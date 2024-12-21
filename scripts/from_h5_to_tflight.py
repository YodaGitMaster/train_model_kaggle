import tensorflow as tf

# Path to the saved Keras model
model_path = "best_model.h5"
tflite_model_path = "best_model.tflite"

# Load the Keras model
model = tf.keras.models.load_model(model_path)

# Convert the Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at {tflite_model_path}")
