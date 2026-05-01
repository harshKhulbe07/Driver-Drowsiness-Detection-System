import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import time


# Step 1: Build model
def build_mobilenet_skeleton():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None
    )
    
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

# Step 2: Create and load
model = build_mobilenet_skeleton()
model.load_weights('eye_weights.weights.h5')
print("✓ Model loaded successfully!")

# Step 3: Create test image
test_image = np.random.rand(96, 96, 3).astype(np.float32) * 255
test_image = np.expand_dims(test_image, axis=0)

# Step 4: Make prediction
prediction = model.predict(test_image, verbose=0)
prob = prediction[0][0]

print(f"Prediction: {prob:.4f}")
print(f"Classification: {'CLOSED' if prob > 0.5 else 'OPEN'}")

# Step 5: Verify output
assert 0 <= prob <= 1, "Probability should be between 0 and 1!"
print("✓ Output valid!")