# predict_image.py
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# EMNIST ByClass label map (0–61)
CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    img = ImageOps.invert(img)                 # invert black/white
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

model = load_model("model_unified.h5")
print("✅ Model loaded successfully (Digits + Letters).")

image_path = input("Enter image path (e.g., sample.png): ").strip()

try:
    data = preprocess_image(image_path)
    prediction = model.predict(data)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_char = CHAR_MAP[pred_class] if pred_class < len(CHAR_MAP) else "?"
    print(f"\n🧩 Predicted Character: {predicted_char}")
    print(f"📈 Confidence: {confidence:.2f}%")
except Exception as e:
    print("⚠️ Error:", e)
