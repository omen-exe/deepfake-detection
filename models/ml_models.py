from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle

def train_ml_models(X_train, y_train):
    # Train different models
    models = {
        "svm": SVC(probability=True),
        "rf": RandomForestClassifier(),
        "lr": LogisticRegression(),
        "nb": GaussianNB(),
    }
    
    trained_models = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Save the model to a file for prediction later
        with open(f"evaluated_models/{model_name}_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        
    return trained_models