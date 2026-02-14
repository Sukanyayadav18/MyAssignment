import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle
import os

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the heart disease dataset"""
        try:
            df = pd.read_csv(data_path)
            
            # Handle missing values
            df = df.dropna()
            
            # Separate features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train(self, data_path):
        """Train the KNN model"""
        X, y = self.load_and_preprocess_data(data_path)
        
        if X is None or y is None:
            return False
            
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
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
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Make probability predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']

def main():
    # Initialize and train the model
    knn_model = KNNModel(n_neighbors=5)
    
    # Train the model
    data_path = r"C:\Users\Asus\heart.csv"
    metrics = knn_model.train(data_path)
    
    if metrics:
        print("KNN Model Training Completed!")
        print("Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Save the model
        knn_model.save_model('model/knn_model.pkl')
        print("Model saved successfully!")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
