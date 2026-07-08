import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "merged_dataset"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "merged_dataset"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "test_dataset"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32
)

def fft_magnitude(image):
    # Convert to float32
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Compute FFT per channel
    fft = tf.signal.fft2d(tf.cast(image, tf.complex64))
    fft_shift = tf.signal.fftshift(fft)
    magnitude = tf.abs(fft_shift)

    # Apply log scaling only (no normalization)
    magnitude = tf.math.log1p(magnitude)

    # Convert RGB FFT → grayscale FFT
    magnitude = tf.reduce_mean(magnitude, axis=-1, keepdims=True)

    return magnitude

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

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

train_fft = train_ds.map(lambda x, y: (fft_magnitude(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_fft = val_ds.map(lambda x, y: (fft_magnitude(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_fft = test_ds.map(lambda x, y: (fft_magnitude(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_fft = train_fft.prefetch(tf.data.AUTOTUNE)
val_fft = val_fft.prefetch(tf.data.AUTOTUNE)
test_fft = test_fft.prefetch(tf.data.AUTOTUNE)

train_dct = train_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_dct   = val_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_dct  = test_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_dct = train_dct.prefetch(tf.data.AUTOTUNE)
val_dct   = val_dct.prefetch(tf.data.AUTOTUNE)
test_dct  = test_dct.prefetch(tf.data.AUTOTUNE)

train_srm = train_ds.map(lambda x, y: (srm_residual_batch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_srm   = val_ds.map(lambda x, y: (srm_residual_batch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_srm  = test_ds.map(lambda x, y: (srm_residual_batch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_srm = train_srm.prefetch(tf.data.AUTOTUNE)
val_srm   = val_srm.prefetch(tf.data.AUTOTUNE)
test_srm  = test_srm.prefetch(tf.data.AUTOTUNE)

# Load base models and build feature extractors
rgb_model = load_model("/home/justincb/models/best_rgb_model_merged.h5")
rgb_extractor = tf.keras.Model(
    inputs=rgb_model.layers[0].input,
    outputs=rgb_model.get_layer("feature_vector").output
)

fft_model = load_model("/home/justincb/models/best_fft_model_merged.h5")
fft_extractor = tf.keras.Model(
    inputs=fft_model.layers[0].input,
    outputs=fft_model.get_layer("fft_feature_vector").output
)

dct_model = load_model("/home/justincb/models/best_dct_model_merged.h5")
dct_extractor = tf.keras.Model(
    inputs=dct_model.layers[0].input,
    outputs=dct_model.get_layer("dct_feature_vector").output
)

srm_model = load_model("/home/justincb/models/best_srm_model_merged.h5")
srm_extractor = tf.keras.Model(
    inputs=srm_model.layers[0].input,
    outputs=srm_model.get_layer("srm_feature_vector").output
)

# Helper: extract embeddings from a tf.data.Dataset
def extract_embeddings(extractor, dataset):
    all_emb = []
    all_y = []
    for batch_x, batch_y in dataset:
        emb = extractor(batch_x, training=False)  # (batch, feat_dim)
        all_emb.append(emb.numpy())
        all_y.append(batch_y.numpy())
    return np.vstack(all_emb), np.hstack(all_y)

# Extract embeddings for train / val / test
# RGB uses raw datasets
X_rgb_train, y_train = extract_embeddings(rgb_extractor, train_ds)
X_rgb_val,   y_val   = extract_embeddings(rgb_extractor, val_ds)
X_rgb_test,  y_test  = extract_embeddings(rgb_extractor, test_ds)

# FFT uses fft datasets
X_fft_train, _ = extract_embeddings(fft_extractor, train_fft)
X_fft_val,   _ = extract_embeddings(fft_extractor, val_fft)
X_fft_test,  _ = extract_embeddings(fft_extractor, test_fft)

# DCT uses dct datasets
X_dct_train, _ = extract_embeddings(dct_extractor, train_dct)
X_dct_val,   _ = extract_embeddings(dct_extractor, val_dct)
X_dct_test,  _ = extract_embeddings(dct_extractor, test_dct)

# SRM uses srm datasets
X_srm_train, _ = extract_embeddings(srm_extractor, train_srm)
X_srm_val,   _ = extract_embeddings(srm_extractor, val_srm)
X_srm_test,  _ = extract_embeddings(srm_extractor, test_srm)

# Concatenate embeddings
X_train = np.concatenate([X_rgb_train, X_fft_train, X_dct_train, X_srm_train], axis=1)
X_val   = np.concatenate([X_rgb_val,   X_fft_val,   X_dct_val,   X_srm_val],   axis=1)
X_test  = np.concatenate([X_rgb_test,  X_fft_test,  X_dct_test,  X_srm_test],  axis=1)

# Function to build the Meta Learner model
def build_meta_learner(input_dim, hidden_units=[128, 64], dropout_rate=0.3, lr=0.001):
    inputs = tf.keras.Input(shape=(input_dim,), name="meta_inputs")

    x = inputs
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="meta_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="meta_learner")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model

# Hyperparameter space
param_grid = [
    # Small
    {"lr": 0.0005, "dropout": 0.2, "hidden_units": [64, 32]},
    {"lr": 0.001, "dropout": 0.2, "hidden_units": [64, 32]},
    {"lr": 0.0005, "dropout": 0.3, "hidden_units": [64, 32]},
    {"lr": 0.001, "dropout": 0.3, "hidden_units": [64, 32]},

    # Large
    {"lr": 0.0005, "dropout": 0.2, "hidden_units": [128, 64]},
    {"lr": 0.001, "dropout": 0.2, "hidden_units": [128, 64]},
    {"lr": 0.0005, "dropout": 0.3, "hidden_units": [128, 64]},
    {"lr": 0.001, "dropout": 0.3, "hidden_units": [128, 64]},
]

results_file = "/home/justincb/models/hparam_results_merged_ensemble.csv"
results = []

best_auc = -1
best_model_path = "/home/justincb/models/best_ensemble_model_merged.h5"

# Training loop for hyperparameter tuning
for i, params in enumerate(param_grid):
    print(f"\n===== Running model {i+1}/{len(param_grid)} =====")
    print(params)

    model = build_meta_learner(
        input_dim=X_train.shape[1],
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout"],
        lr=params["lr"]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Log results
    results.append({
        "lr": params["lr"],
        "hidden_units": params["hidden_units"],
        "dropout": params["dropout"],
        "val_accuracy": history.history["val_accuracy"][-1],
        "val_auc": history.history["val_auc"][-1],
        "test_accuracy": acc,
        "test_auc": auc
    })

    # Save best model
    if auc > best_auc:
        best_auc = auc
        model.save(best_model_path)
        print(f"New best model saved with AUC {auc:.4f}")

df = pd.DataFrame(results)
df.to_csv(results_file, index=False)

print("\nHyperparameter tuning complete.")
print(f"Best model AUC: {best_auc:.4f}")
print(f"Results saved to: {results_file}")
print(f"Best model saved to: {best_model_path}")