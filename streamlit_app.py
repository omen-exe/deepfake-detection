import streamlit as st
from PIL import Image
from models.utils import preprocess_image
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the models
svm_model = joblib.load('evaluated_models/svm_model.pkl')
rf_model = joblib.load('evaluated_models/rf_model.pkl')
lr_model = joblib.load('evaluated_models/lr_model.pkl')
nb_model = joblib.load('evaluated_models/nb_model.pkl')
cnn_model = load_model('evaluated_models/cnn_final_model.h5')

# Model list
ml_models = {
    'SVM': svm_model,
    'Random Forest': rf_model,
    'Logistic Regression': lr_model,
    'Naive Bayes': nb_model
}


def preprocess_image_for_prediction(img_path, is_cnn_model=False):
    # Load image, resize it to (224, 224), and ensure it has 3 channels
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)  # Convert the image to a numpy array

    if is_cnn_model:
        # For CNN, we need the image to be in shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
    else:
        # For ML models, we need to flatten the image to shape (1, 224*224*3)
        img_array = img_array.flatten()  # Flatten to 1D array (1, 224*224*3)
    
    return img_array


def predict_all_models(img_path, cnn_model, ml_models):
    img_cnn = preprocess_image_for_prediction(img_path, is_cnn_model=True)  # For CNN model
    cnn_pred = cnn_model.predict(img_cnn)
    
    # Prepare the image for ML models (flattened image)
    img_ml = preprocess_image_for_prediction(img_path, is_cnn_model=False)  # For ML models
    
    # Predictions for ML models
    ml_predictions = {}
    for model_name, model in ml_models.items():
        pred = model.predict([img_ml])  # Make prediction with flattened image
        ml_predictions[model_name] = pred[0]  # Store the prediction

    return ml_predictions, cnn_pred


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)


# Streamlit app logic
st.title("Deepfake Detection")
st.write("Upload an image to classify it as real or fake.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a temporary path
    img_path = 'uploaded_image.jpg'  # Temporary save
    img.save(img_path)

    # Evaluate the image with all models
    ml_predictions, cnn_prediction = predict_all_models(img_path, cnn_model, ml_models)

    # Display results
    st.write("### ML Model Predictions:")
    ml_y_true = [0] * len(ml_models)  # Assume the image is real (0) for all models (you can modify this if you know the actual label)
    ml_y_pred = []

    for model_name, pred in ml_predictions.items():
        result = "Fake" if pred == 1 else "Real"
        st.write(f"{model_name}: {result}")
        ml_y_pred.append(pred)

    # Display CNN model result
    cnn_result = "Fake" if cnn_prediction == 1 else "Real"
    st.write(f"### CNN Model Prediction: {cnn_result}")
