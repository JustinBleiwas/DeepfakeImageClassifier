# Import modules
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "dataset_1_cleaned"),
    labels="inferred",
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(os.environ["SLURM_TMPDIR"], "dataset_1_cleaned"),
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

train_dct = train_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_dct   = val_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_dct  = test_ds.map(lambda x, y: (dct_blockwise_residual(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Overlap CPU and GPU work to improve runtime
train_dct = train_dct.prefetch(tf.data.AUTOTUNE)
val_dct   = val_dct.prefetch(tf.data.AUTOTUNE)
test_dct  = test_dct.prefetch(tf.data.AUTOTUNE)

# Compute class weights
train_root = os.path.join(os.environ["SLURM_TMPDIR"], "dataset_1_cleaned")

class_names = sorted(os.listdir(train_root))
counts = [len(os.listdir(os.path.join(train_root, cls))) for cls in class_names]

classes = np.arange(len(class_names))
weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.repeat(classes, counts))

class_weights = {cls: w for cls, w in zip(classes, weights)}

# Function to build the DCT model
def build_dct_cnn(input_shape=(256, 256, 1), conv_filters=[32, 64, 128, 256], dense_units=128, dropout_rate=0.3):
    model = models.Sequential([layers.Input(shape=input_shape)])

    # Convolution blocks
    for f in conv_filters:
        model.add(layers.Conv2D(f, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Hyperparameter space
param_grid = [
    # Take the top 2 best AUC models from the refined grid search and the best model from the base grid search
    # Then add a few more model around those to ensure the best final AUC

    # As the number of filters is constant between all best models, this search will focus on learning rate and dropout rate

    # Group 1: Refined Grid Best AUC Space
    {"lr": 0.0005, "dropout": 0.2, "conv_filters": [32, 64, 128, 256]}, # Refined Grid best AUC
    {"lr": 0.00025, "dropout": 0.2, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.001, "dropout": 0.2, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0005, "dropout": 0.15, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0005, "dropout": 0.25, "conv_filters": [32, 64, 128, 256]},

    # Group 2: Refined Grid Second Best AUC Space
    {"lr": 0.0001, "dropout": 0.4, "conv_filters": [32, 64, 128, 256]}, # Refined Grid second best AUC
    {"lr": 0.00005, "dropout": 0.4, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0002, "dropout": 0.4, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.35, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.45, "conv_filters": [32, 64, 128, 256]},

    # Group 3: Base Grid Best AUC Space
    {"lr": 0.0001, "dropout": 0.3, "conv_filters": [32, 64, 128, 256]}, # Base Grid best AUC
    {"lr": 0.00005, "dropout": 0.3, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0002, "dropout": 0.3, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.25, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.35, "conv_filters": [32, 64, 128, 256]},
]

results_file = "/home/justincb/models/final_results_D1_dct.csv"
results = []

best_auc = -1
best_model_path = "/home/justincb/models/best_dct_model_D1.h5"

# Training loop for hyperparameter tuning
for i, params in enumerate(param_grid):
    print(f"\n===== Running model {i+1}/{len(param_grid)} =====")
    print(params)

    model = build_dct_cnn(
        conv_filters=params["conv_filters"],
        dropout_rate=params["dropout"]
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
        train_dct,
        validation_data=val_dct,
        epochs=20,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    loss, acc, auc = model.evaluate(test_dct, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Log results
    best_epoch = np.argmin(history.history["val_loss"])

    results.append({
        "lr": params["lr"],
        "conv_filters": params["conv_filters"],
        "dropout": params["dropout"],
        "val_accuracy": history.history["val_accuracy"][best_epoch],
        "val_auc": history.history["val_auc"][best_epoch],
        "test_accuracy": acc,
        "test_auc": auc
    })

    # Save best model
    if auc > best_auc:
        best_auc = auc
        model.save(best_model_path)
        print(f"New best model saved with AUC {auc:.4f}")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(results_file, index=False)

print("\nHyperparameter tuning complete.")
print(f"Best model AUC: {best_auc:.4f}")
print(f"Results saved to: {results_file}")
print(f"Best model saved to: {best_model_path}")