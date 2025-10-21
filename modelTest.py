from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import math

#load model
model = load_model("notebooks/distracted_driver_detection.keras")

#class names
class_names = ["Safe driving", "Distracted", "Tired"]

#folder
folder = "C:/Users/lurpd/Documents/Development/Datasets/MyData/"
target_size = (128, 128)

#get images from folder
image_ids = [i for i in range(30, 63) if os.path.exists(os.path.join(folder, f"{i}.png"))]

#display grid
cols = 8
rows = math.ceil(len(image_ids) / cols)

max_width_px = 2500
max_height_px = 1300
dpi = 100

fig_width = min(max_width_px / dpi, cols * 3)
fig_height = min(max_height_px / dpi, rows * 3)

plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

#image process/display
for idx, i in enumerate(image_ids):
    img_path = os.path.join(folder, f"{i}.png")

    #preprocess
    img = Image.open(img_path).convert("L")
    img = ImageOps.pad(img, target_size, color=0, method=Image.Resampling.LANCZOS)
    img_array = np.expand_dims(np.array(img).astype(np.float32) / 255.0, axis=(0, -1))

    #predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]
    confidence = np.max(predictions) * 100

    #display
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.text(
        0.5, -0.05, 
        f"{i}.png\n{predicted_label} ({confidence:.1f}%)",
        fontsize=9,
        ha="center", va="top", transform=plt.gca().transAxes
    )

plt.tight_layout()
plt.show()
