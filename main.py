import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from models.ml_models import train_ml_models
from models.cnn_model import create_cnn_model
from models.utils import preprocess_image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def is_image_file(file_name):
    """Check if the file has an allowed image extension."""
    return any(file_name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    X = []  
    y = [] 

    for folder in ['real', 'fake']:
        label = 0 if folder == 'real' else 1
        for img_name in os.listdir(f'data/{folder}'):
            if is_image_file(img_name):
                img_path = f'data/{folder}/{img_name}'
                X.append(preprocess_image(img_path).flatten())  # Flatten image for ML models
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


def train_and_evaluate_ml_models(X_train, y_train, X_test, y_test):
    """Train and evaluate ML models."""
    ml_models = train_ml_models(X_train, y_train)

    for model_name, model in ml_models.items():
        print(f"Evaluating {model_name}:")

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Print the metrics
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC score: {roc_auc:.4f}")

        
        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(y_test, y_prob, model_name)
        plot_precision_recall_curve(y_test, y_prob, model_name)


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.show()


def train_and_evaluate_cnn(X_train, y_train, X_test, y_test):
    # Train the CNN model
    X_train_cnn = X_train.reshape(-1, 224, 224, 3)  # Reshape for CNN
    X_test_cnn = X_test.reshape(-1, 224, 224, 3)
    
    cnn_model = create_cnn_model()
    checkpoint = ModelCheckpoint('evaluated_models/cnn_best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    cnn_model.fit(X_train_cnn, y_train, epochs=10, validation_data=(X_test_cnn, y_test), callbacks=[checkpoint])
    
    # Save the CNN model
    cnn_model.save('evaluated_models/cnn_final_model.h5')
    
    # Make predictions on the test data
    cnn_pred_prob = cnn_model.predict(X_test_cnn)
    
    if cnn_pred_prob.shape[1] == 1:  # Single output for binary classification
        cnn_pred_prob_positive = cnn_pred_prob[:, 0]  # Probability of the positive class
    else:
        cnn_pred_prob_positive = cnn_pred_prob[:, 1]  # If softmax is used
    
    # Convert predictions to class labels
    cnn_pred = (cnn_pred_prob_positive >= 0.5).astype(int)  # Thresholding at 0.5
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, cnn_pred)
    precision = precision_score(y_test, cnn_pred, average='binary')
    recall = recall_score(y_test, cnn_pred, average='binary')
    f1 = f1_score(y_test, cnn_pred, average='binary')
    roc_auc = roc_auc_score(y_test, cnn_pred_prob_positive)
    
    # Print metrics
    print(f"Metrics for CNN Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Confusion Matrix for CNN
    cm = confusion_matrix(y_test, cnn_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    ax.set_title('CNN Model - Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()

    plot_roc_curve(y_test, cnn_pred_prob_positive, "CNN")
    plot_precision_recall_curve(y_test, cnn_pred_prob_positive, "CNN")


def main():
    os.makedirs('evaluated_models', exist_ok=True)
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Define different test sizes
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    for test_size in test_sizes:
        print(f"Evaluating on test-size split of : {test_size}")
        # Split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Train and evaluate ML models
        train_and_evaluate_ml_models(X_train, y_train, X_test, y_test)

        # Train and evaluate CNN model
        train_and_evaluate_cnn(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()