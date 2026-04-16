import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = './models/final_model.keras'
TEST_IMAGE = './project6/Drone/drone1.png'

def test_inference():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Loading test image from {TEST_IMAGE}...")
    image = Image.open(TEST_IMAGE).convert('RGB').resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "DRONE" if prediction > 0.5 else "BIRD"
    confidence = prediction if prediction > 0.5 else 1.0 - prediction
    
    print("-" * 30)
    print(f"RESULT: {label}")
    print(f"CONFIDENCE: {confidence:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    test_inference()
