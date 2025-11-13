# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

print("🔄 Downloading EMNIST dataset (byclass) using TensorFlow Datasets...")

# Load EMNIST "byclass" split
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
)

num_classes = ds_info.features["label"].num_classes
print(f"✅ Dataset loaded with {num_classes} classes")

# Convert to NumPy arrays for training
def to_numpy(ds):
    images, labels = [], []
    for img, label in tfds.as_numpy(ds):
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = to_numpy(ds_train)
X_test, y_test = to_numpy(ds_test)

# Normalize and reshape
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("🚀 Training model... (this will take a few minutes)")
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Model accuracy: {acc*100:.2f}%")

# Save model
model.save("model_unified.h5")
print("💾 Model saved successfully as model_unified.h5")
