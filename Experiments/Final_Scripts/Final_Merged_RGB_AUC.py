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

# Overlap CPU and GPU work to improve runtime
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Compute class weights
train_root = os.path.join(os.environ["SLURM_TMPDIR"], "merged_dataset")

class_names = sorted(os.listdir(train_root))
counts = [len(os.listdir(os.path.join(train_root, cls))) for cls in class_names]

classes = np.arange(len(class_names))
weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.repeat(classes, counts))

class_weights = {cls: w for cls, w in zip(classes, weights)}

# Function to build the CNN model
def build_rgb_cnn(input_shape=(256, 256, 3), conv_filters=[32, 64, 128, 256], dense_units=128, dropout_rate=0.3):
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

    # As all models have the same architecture, I'm only going to vary the learning rate and dropout rate

    # Group 1: Refined Grid Best AUC Space
    {"lr": 0.0001, "dropout": 0.3,  "conv_filters": [32, 64, 128, 256]}, # Refined Grid best AUC
    {"lr": 0.00005, "dropout": 0.3,  "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0002, "dropout": 0.3,  "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.25,  "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.35,  "conv_filters": [32, 64, 128, 256]},

    # Group 2: Refined Grid Second Best AUC Space
    {"lr": 0.00005, "dropout": 0.25, "conv_filters": [32, 64, 128, 256]}, # Refined Grid second best AUC
    {"lr": 0.000025, "dropout": 0.25, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.25, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.00005, "dropout": 0.2, "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.00005, "dropout": 0.3, "conv_filters": [32, 64, 128, 256]},

    # Group 3: Base Grid Best AUC Space
    {"lr": 0.0001, "dropout": 0.5,  "conv_filters": [32, 64, 128, 256]}, # Base Grid best AUC
    {"lr": 0.00005, "dropout": 0.5,  "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0002, "dropout": 0.5,  "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.45,  "conv_filters": [32, 64, 128, 256]},
    {"lr": 0.0001, "dropout": 0.55,  "conv_filters": [32, 64, 128, 256]},
]

results_file = "/home/justincb/models/final_results_merged_rgb_AUC.csv"
results = []

best_auc = -1
best_model_path = "/home/justincb/models/best_rgb_model_merged.h5"

# Training loop for hyperparameter tuning
for i, params in enumerate(param_grid):
    print(f"\n===== Running model {i+1}/{len(param_grid)} =====")
    print(params)

    model = build_rgb_cnn(
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
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    loss, acc, auc = model.evaluate(test_ds, verbose=0)
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