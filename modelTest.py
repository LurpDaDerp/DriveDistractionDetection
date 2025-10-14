from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

#load model
model = load_model("notebooks/distracted_driver_detection.keras") 

#make class names (c0–c9 from training data)
class_names = [
    "Safe driving",                 
    "Texting - right hand",          
    "Talking on phone - right hand", 
    "Texting - left hand",           
    "Talking on phone - left hand",  
    "Operating the radio",           
    "Drinking",                      
    "Reaching behind",               
    "Hair and makeup",               
    "Talking to passenger"           
]

#image 
img_path = "C:/Users/lurpd/Downloads/12.jpg"

#process image
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

#predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = class_names[predicted_class]

#print results to terminal
print("Predicted class index:", predicted_class)
print("Predicted label:", predicted_label)

#show image and results in window
plt.imshow(image.load_img(img_path))
plt.axis("off")
plt.title(f"Prediction: {predicted_label}")
plt.show()
