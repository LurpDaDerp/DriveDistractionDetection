from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#load model
model = load_model("notebooks/distracted_driver_detection.keras")

#class names
class_names = [
    "Safe driving", "Texting - right hand", "Talking on phone - right hand",
    "Texting - left hand", "Talking on phone - left hand", "Operating the radio",
    "Drinking", "Reaching behind", "Hair and makeup", "Talking to passenger"
]

#load image
img_path = "C:/Users/lurpd/Documents/Development/Datasets/MyData/39.png"

target_size = (128, 128)
img = Image.open(img_path).convert("RGB")

img = ImageOps.pad(img, target_size, color=(0, 0, 0), method=Image.Resampling.LANCZOS)

#normalize
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 127.0 

#run model
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = class_names[predicted_class]

#output
print("Predicted class index:", predicted_class)
print("Predicted label:", predicted_label)

#image display
plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {predicted_label}")
plt.show()
