"""Train a mask / no-mask classifier. Expects dataset/with_mask and dataset/without_mask."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG = 224
train = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="training", seed=42, image_size=(IMG, IMG), batch_size=32)
val = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="validation", seed=42, image_size=(IMG, IMG), batch_size=32)
print("Classes:", train.class_names)  # alphabetical: with_mask=0, without_mask=1

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
base.trainable = False
model = models.Sequential([
    layers.Rescaling(1 / 127.5, offset=-1),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=5)
model.save("mask_detector.keras")
