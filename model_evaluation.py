import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import model classes
from model.knn import KNNModel
from model.logistic_regression import LogisticRegressionModel
from model.decision_tree import DecisionTreeModel
from model.naive_bayes import NaiveBayesModel
from model.random_forest import RandomForestModel
from model.xgboost_model import XGBoostModel

class ModelEvaluator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {
            'K-Nearest Neighbors': KNNModel(),
            'Logistic Regression': LogisticRegressionModel(),
            'Decision Tree': DecisionTreeModel(),
            'Naive Bayes': NaiveBayesModel(),
            'Random Forest': RandomForestModel(),
            'XGBoost': XGBoostModel()
        }
        self.metrics_df = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        X = df.drop('target', axis=1)
        y = df['target']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def train_all_models(self):
        """Train all models and collect metrics"""
        print("Training all models...")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            try:
                metrics = model.train(self.data_path)
                if metrics:
                    results[model_name] = metrics
                    print(f"✓ {model_name} trained successfully")
                else:
                    print(f"✗ {model_name} training failed")
            except Exception as e:
                print(f"✗ {model_name} training failed with error: {e}")
        
        # Create DataFrame with results
        self.metrics_df = pd.DataFrame(results).T
        return self.metrics_df
    
    def create_comparison_table(self):
        """Create a formatted comparison table"""
        if self.metrics_df is None:
            self.train_all_models()
        
        # Round values for better presentation
        comparison_df = self.metrics_df.round(4)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string())
        print("="*80)
        
        return comparison_df
    
    def plot_model_comparison(self):
        """Create visualization comparing model performances"""
        if self.metrics_df is None:
            self.train_all_models()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score', 'mcc_score']
        metric_titles = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row = i // 3
            col = i % 3
            
            ax = axes[row, col]
            values = self.metrics_df[metric].sort_values(ascending=False)
            
            bars = ax.bar(range(len(values)), values.values, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(values))))
            
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        X_train, X_test, y_train, y_test = self.load_data()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        
        model_names = list(self.models.keys())
        
        for i, (model_name, model) in enumerate(self.models.items()):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            try:
                # Make predictions
                if hasattr(model, 'scaler') and model.scaler is not None:
                    X_test_processed = model.scaler.transform(X_test)
                else:
                    X_test_processed = X_test
                
                y_pred = model.model.predict(X_test_processed)
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Plot
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           cbar_kws={'shrink': 0.8})
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{model_name} - Error', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """Generate a detailed evaluation report"""
        if self.metrics_df is None:
            self.train_all_models()
        
        print("\n" + "="*100)
        print("DETAILED MODEL EVALUATION REPORT")
        print("="*100)
        
        # Best performing model for each metric
        print("\nBEST PERFORMING MODELS BY METRIC:")
        print("-" * 40)
        
        for metric in self.metrics_df.columns:
            best_model = self.metrics_df[metric].idxmax()
            best_score = self.metrics_df[metric].max()
            print(f"{metric.upper():<20}: {best_model:<25} ({best_score:.4f})")
        
        # Overall best model (based on average performance)
        avg_performance = self.metrics_df.mean(axis=1)
        overall_best = avg_performance.idxmax()
        print(f"\nOVERALL BEST MODEL: {overall_best} (Average Score: {avg_performance[overall_best]:.4f})")
        
        # Model rankings
        print(f"\nMODEL RANKINGS (by average performance):")
        print("-" * 50)
        rankings = avg_performance.sort_values(ascending=False)
        for i, (model, score) in enumerate(rankings.items(), 1):
            print(f"{i}. {model:<25}: {score:.4f}")
        
        # Performance insights
        print(f"\nPERFORMANCE INSIGHTS:")
        print("-" * 30)
        
        # Best accuracy
        best_acc_model = self.metrics_df['accuracy'].idxmax()
        best_acc_score = self.metrics_df['accuracy'].max()
        print(f"• Highest Accuracy: {best_acc_model} ({best_acc_score:.4f})")
        
        # Best AUC
        best_auc_model = self.metrics_df['auc_score'].idxmax()
        best_auc_score = self.metrics_df['auc_score'].max()
        print(f"• Highest AUC Score: {best_auc_model} ({best_auc_score:.4f})")
        
        # Best F1
        best_f1_model = self.metrics_df['f1_score'].idxmax()
        best_f1_score = self.metrics_df['f1_score'].max()
        print(f"• Highest F1 Score: {best_f1_model} ({best_f1_score:.4f})")
        
        # Model stability (lowest std across metrics)
        model_std = self.metrics_df.std(axis=1)
        most_stable = model_std.idxmin()
        print(f"• Most Stable Model: {most_stable} (Std: {model_std[most_stable]:.4f})")
        
        print("="*100)
        
        return self.metrics_df

def main():
    # Initialize evaluator
    data_path = r"C:\Users\Asus\heart.csv"
    evaluator = ModelEvaluator(data_path)
    
    # Train all models and get metrics
    print("Starting comprehensive model evaluation...")
    metrics_df = evaluator.train_all_models()
    
    # Create comparison table
    comparison_table = evaluator.create_comparison_table()
    
    # Generate detailed report
    evaluator.generate_detailed_report()
    
    # Create visualizations
    evaluator.plot_model_comparison()
    evaluator.plot_confusion_matrices()
    
    # Save results to CSV
    comparison_table.to_csv('model_comparison_results.csv')
    print(f"\nResults saved to 'model_comparison_results.csv'")
    
    print("\nModel evaluation completed successfully!")

if __name__ == "__main__":
    main()
