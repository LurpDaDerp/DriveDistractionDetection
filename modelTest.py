from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#load model
model = load_model("notebooks/distracted_driver_detection.keras")

#class names
class_names = [
    "Safe driving", 
    "Operating the radio",
    "Drinking", 
    "Reaching behind", 
    "Hair and makeup", 
    "Talking to passenger",
    "General Distracted",
    "Sleepy",
    "Yawn"
]

#load and preprocess image
img_path = "C:/Users/lurpd/Documents/Development/Datasets/MyData/49.png"
target_size = (128, 128)

#open image in grayscale
img = Image.open(img_path).convert("L")

#letterbox
img = ImageOps.pad(img, target_size, color=0, method=Image.Resampling.LANCZOS)

#convert to NumPy and normalize
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=(0, -1))  

#prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = class_names[predicted_class]

#output
print("Predicted class index:", predicted_class)
print("Predicted label:", predicted_label)

#display for debug
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title(f"Prediction: {predicted_label}")
plt.show()
