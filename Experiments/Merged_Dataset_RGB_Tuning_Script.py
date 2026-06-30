# Import modules
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "/scratch/justincb/merged_dataset",
    labels="inferred",
    image_size=(256, 256),
    batch_size=64,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "/scratch/justincb/merged_dataset",
    labels="inferred",
    image_size=(256, 256),
    batch_size=64,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "/scratch/justincb/test_dataset",
    labels="inferred",
    image_size=(256, 256),
    batch_size=64
)

# Overlap CPU and GPU work to improve runtime
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Compute class weights
y_train = np.concatenate([y.numpy() for _, y in train_ds])
weights = compute_class_weight(class_weight='balanced',
                               classes=np.unique(y_train),
                               y=y_train)
class_weights = {cls: w for cls, w in zip(np.unique(y_train), weights)}

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
    # Light
    {"lr": 1e-4, "dropout": 0.2, "conv_filters": [16, 32, 64, 128]},
    {"lr": 5e-4, "dropout": 0.2, "conv_filters": [16, 32, 64, 128]},

    # Medium (baseline)
    {"lr": 1e-4, "dropout": 0.3, "conv_filters": [32, 64, 128, 256]},
    {"lr": 5e-4, "dropout": 0.3, "conv_filters": [32, 64, 128, 256]},

    # Heavy
    {"lr": 1e-4, "dropout": 0.4, "conv_filters": [64, 128, 256, 512]},
    {"lr": 5e-4, "dropout": 0.4, "conv_filters": [64, 128, 256, 512]},

    # Mixture
    {"lr": 1e-4, "dropout": 0.3, "conv_filters": [32, 64, 64, 128]},
    {"lr": 1e-4, "dropout": 0.5, "conv_filters": [32, 64, 128, 256]},
]

results_file = "/home/justincb/models/hparam_results_merged_rgb.csv"
results = []

best_acc = -1
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
    results.append({
        "lr": params["lr"],
        "conv_filters": params["conv_filters"],
        "dropout": params["dropout"],
        "val_accuracy": history.history["val_accuracy"][-1],
        "val_auc": history.history["val_auc"][-1],
        "test_accuracy": acc,
        "test_auc": auc
    })

    # Save best model
    if acc > best_acc:
        best_acc = acc
        model.save(best_model_path)
        print(f"New best model saved with accuracy {acc:.4f}")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(results_file, index=False)

print("\nHyperparameter tuning complete.")
print(f"Best model accuracy: {best_acc:.4f}")
print(f"Results saved to: {results_file}")
print(f"Best model saved to: {best_model_path}")