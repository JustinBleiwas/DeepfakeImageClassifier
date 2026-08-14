import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

# Load datasets
fifty_percent_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "bluesky_faces_cleaned_50_percent"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32
)

seventy_five_percent_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "bluesky_faces_cleaned_75_percent"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32
)

one_hundred_percent_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "bluesky_faces_cleaned_100_percent"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32
)

# 8×8 orthonormal DCT matrix
def dct_matrix(N=8):
    k = tf.range(N, dtype=tf.float32)
    n = tf.range(N, dtype=tf.float32)
    k = tf.reshape(k, (-1, 1))
    n = tf.reshape(n, (1, -1))

    pi = tf.constant(np.pi, dtype=tf.float32)

    mat = tf.cos((pi / N) * (n + 0.5) * k)
    mat = mat * tf.sqrt(2.0 / N)

    # Replace first row with sqrt(1/N)
    mat = tf.tensor_scatter_nd_update(
        mat,
        indices=[[0]],
        updates=[tf.fill([N], tf.sqrt(1.0 / N))]
    )

    return mat

DCT8 = dct_matrix(8)
IDCT8 = tf.transpose(DCT8)  # inverse for orthonormal basis

def dct_blockwise_residual(batch):
    # Convert to float32 in [0,1]
    batch = tf.image.convert_image_dtype(batch, tf.float32)

    # Split RGB → process each channel independently
    channels = tf.split(batch, num_or_size_splits=batch.shape[-1], axis=-1)
    residual_channels = []

    for ch in channels:
        # Extract 8×8 blocks
        patches = tf.image.extract_patches(
            images=ch,
            sizes=[1, 8, 8, 1],
            strides=[1, 8, 8, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        B = tf.shape(patches)[0]
        H_blocks = tf.shape(patches)[1]
        W_blocks = tf.shape(patches)[2]

        # reshape to (B, H_blocks, W_blocks, 8, 8)
        patches = tf.reshape(patches, (B, H_blocks, W_blocks, 8, 8))

        # 2D DCT: DCT8 * block * DCT8^T
        dct_blocks = tf.einsum('ij, bhwij -> bhwij', DCT8, patches)
        dct_blocks = tf.einsum('bhwij, ij -> bhwij', dct_blocks, tf.transpose(DCT8))

        # ----- Per‑block stabilization -----
        # scale by block energy (not global normalization)
        energy = tf.reduce_mean(tf.abs(dct_blocks), axis=[3, 4], keepdims=True)
        dct_blocks = dct_blocks / (energy + 1e-6)

        # log‑scale to compress dynamic range
        dct_blocks = tf.math.log1p(tf.abs(dct_blocks))

        # ----- Inverse DCT to reconstruct residual block -----
        idct_blocks = tf.einsum('ij, bhwij -> bhwij', IDCT8, dct_blocks)
        idct_blocks = tf.einsum('bhwij, ij -> bhwij', idct_blocks, tf.transpose(IDCT8))

        # reshape back into full image
        H = H_blocks * 8
        W = W_blocks * 8
        idct_blocks = tf.reshape(idct_blocks, (B, H, W, 1))

        residual_channels.append(idct_blocks)

    # Combine channels → (B, H, W, 3)
    residual = tf.concat(residual_channels, axis=-1)

    # Grayscale aggregation → (B, H, W, 1)
    residual = tf.reduce_mean(residual, axis=-1, keepdims=True)

    return residual

def srm_residual_batch(batch):
    # Convert to float32 in [0,1]
    batch = tf.image.convert_image_dtype(batch, tf.float32)

    # SRM Filters
    srm_kernels = tf.constant([
        [[ 0,  0,  0],
         [ 0,  1, -1],
         [ 0, -1,  1]],

        [[ 0,  0,  0],
         [ 1, -2,  1],
         [ 0,  0,  0]],

        [[-1,  2, -1],
         [ 2, -4,  2],
         [-1,  2, -1]]
    ], dtype=tf.float32)

    # Reshape to (3, 3, 1, 3):
    # 3 filters, 3×3 spatial size, 1 input channel, 3 output channel
    srm_kernels = tf.reshape(srm_kernels, (3, 3, 1, 3))

    # Apply SRM filters to each RGB channel independently
    channels = tf.split(batch, num_or_size_splits=batch.shape[-1], axis=-1)
    residuals = []

    for ch in channels:
        # Convolve: (B, H, W, 1) → (B, H, W, 3)
        r = tf.nn.conv2d(ch, srm_kernels, strides=1, padding='SAME')
        residuals.append(r)

    # Concatenate residuals from each channel → (B, H, W, 9)
    residual = tf.concat(residuals, axis=-1)

    # Magnitude + grayscale aggregation
    residual = tf.abs(residual)
    residual = tf.reduce_mean(residual, axis=-1, keepdims=True)  # (B, H, W, 1)

    return residual

fifty_percent_ds = fifty_percent_ds.prefetch(tf.data.AUTOTUNE)
seventy_five_percent_ds = seventy_five_percent_ds.prefetch(tf.data.AUTOTUNE)
one_hundred_percent_ds = one_hundred_percent_ds.prefetch(tf.data.AUTOTUNE)

fifty_percent_dct = fifty_percent_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
seventy_five_percent_dct = seventy_five_percent_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
one_hundred_percent_dct = one_hundred_percent_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)

fifty_percent_srm = fifty_percent_ds.map(lambda x, y: (srm_residual_batch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
seventy_five_percent_srm = seventy_five_percent_ds.map(lambda x, y: (srm_residual_batch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
one_hundred_percent_srm = one_hundred_percent_ds.map(lambda x, y: (srm_residual_batch(x), y), num_parallel_calls=tf.data.AUTOTUNE)

ensemble_model = load_model("/home/justincb/models/best_ensemble_model_merged.h5")
rgb_model = load_model("/home/justincb/models/best_rgb_model_merged.h5")
dct_model = load_model("/home/justincb/models/best_dct_model_merged.h5")
srm_model = load_model("/home/justincb/models/best_srm_model_merged.h5")

def rebuild_with_feature(model, input_shape):
    print(f"\nRebuilding functional graph (layer-by-layer) for input {input_shape}...")
    inp = tf.keras.Input(shape=input_shape)
    x = inp

    dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
    print(f"  Dense layers found: {[l.name for l in dense_layers]}")
    if len(dense_layers) < 2:
        raise ValueError("Model does not contain enough Dense layers.")
    feature_layer = dense_layers[-2]
    print(f"  Selected feature layer: {feature_layer.name}")

    feature_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer is feature_layer:
            feature_tensor = x

    func_model = tf.keras.Model(inputs=inp, outputs=x)
    print("Functional graph rebuilt.")
    return func_model, feature_tensor

print("Rebuild models and capture feature tensors")
rgb_model_func, rgb_feature_tensor = rebuild_with_feature(rgb_model, (256, 256, 3))
dct_model_func, dct_feature_tensor = rebuild_with_feature(dct_model, (256, 256, 1))
srm_model_func, srm_feature_tensor = rebuild_with_feature(srm_model, (256, 256, 1))

print("Build extractors")
rgb_extractor = tf.keras.Model(inputs=rgb_model_func.input, outputs=rgb_feature_tensor)
print("RGB extractor built.")

dct_extractor = tf.keras.Model(inputs=dct_model_func.input, outputs=dct_feature_tensor)
print("DCT extractor built.")

srm_extractor = tf.keras.Model(inputs=srm_model_func.input, outputs=srm_feature_tensor)
print("SRM extractor built.")

print("Test extractor outputs")
dummy_rgb = tf.zeros((1, 256, 256, 3))
dummy_dct = tf.zeros((1, 256, 256, 1))
dummy_srm = tf.zeros((1, 256, 256, 1))

print("RGB:", rgb_extractor(dummy_rgb).shape)
print("DCT:", dct_extractor(dummy_dct).shape)
print("SRM:", srm_extractor(dummy_srm).shape)

# Helper: extract embeddings from a tf.data.Dataset
def extract_embeddings(extractor, dataset):
    all_emb = []
    all_y = []
    for batch_x, batch_y in dataset:
        emb = extractor(batch_x, training=False)  # (batch, feat_dim)
        all_emb.append(emb.numpy())
        all_y.append(batch_y.numpy())
    return np.vstack(all_emb), np.hstack(all_y)

# RGB uses raw datasets
X_rgb_50, y_50 = extract_embeddings(rgb_extractor, fifty_percent_ds)
X_rgb_75, y_75 = extract_embeddings(rgb_extractor, seventy_five_percent_ds)
X_rgb_100, y_100 = extract_embeddings(rgb_extractor, one_hundred_percent_ds)

# DCT uses dct datasets
X_dct_50, _ = extract_embeddings(dct_extractor, fifty_percent_dct)
X_dct_75, _ = extract_embeddings(dct_extractor, seventy_five_percent_dct)
X_dct_100, _ = extract_embeddings(dct_extractor, one_hundred_percent_dct)

# SRM uses srm datasets
X_srm_50, _ = extract_embeddings(srm_extractor, fifty_percent_srm)
X_srm_75, _ = extract_embeddings(srm_extractor, seventy_five_percent_srm)
X_srm_100, _ = extract_embeddings(srm_extractor, one_hundred_percent_srm)

# Concatenate embeddings
X_50 = np.concatenate([X_rgb_50, X_dct_50, X_srm_50], axis=1)
X_75 = np.concatenate([X_rgb_75, X_dct_75, X_srm_75], axis=1)
X_100 = np.concatenate([X_rgb_100, X_dct_100, X_srm_100], axis=1)

def count_fakes(X, model):
    preds = model.predict(X, verbose=0)
    preds = np.round(preds).astype(int)

    # 0 = fake, 1 = real
    fake_count = np.sum(preds == 0)
    total = len(preds)

    return fake_count, total

num_fakes_50, total_50 = count_fakes(X_50, ensemble_model)
num_fakes_75, total_75 = count_fakes(X_75, ensemble_model)
num_fakes_100, total_100 = count_fakes(X_100, ensemble_model)

print("50% margin:")
print(f"  Fake: {num_fakes_50} / {total_50}  ({num_fakes_50/total_50:.2%})")

print("75% margin:")
print(f"  Fake: {num_fakes_75} / {total_75}  ({num_fakes_75/total_75:.2%})")

print("100% margin:")
print(f"  Fake: {num_fakes_100} / {total_100}  ({num_fakes_100/total_100:.2%})")