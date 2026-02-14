import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle
import os

class DecisionTreeModel:
    def __init__(self, random_state=42):
        self.model = DecisionTreeClassifier(random_state=random_state, max_depth=10)
        self.is_trained = False
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the heart disease dataset"""
        try:
            df = pd.read_csv(data_path)
            df = df.dropna()
            X = df.drop('target', axis=1)
            y = df['target']
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train(self, data_path):
        """Train the Decision Tree model"""
        X, y = self.load_and_preprocess_data(data_path)
        
        if X is None or y is None:
            return False
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'mcc_score': matthews_corrcoef(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']

def main():
    dt_model = DecisionTreeModel()
    data_path = r"C:\Users\Asus\heart.csv"
    metrics = dt_model.train(data_path)
    
    if metrics:
        print("Decision Tree Model Training Completed!")
        print("Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        dt_model.save_model('model/decision_tree_model.pkl')
        print("Model saved successfully!")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
