from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    print(f"Processing image at : {img_path}")
    """Preprocess the image to be fed into the model."""
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    
    return img_array.flatten()  # For ML models, flatten the image to 1D