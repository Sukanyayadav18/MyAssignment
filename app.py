import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import model classes
from model.knn import KNNModel
from model.logistic_regression import LogisticRegressionModel
from model.decision_tree import DecisionTreeModel
from model.naive_bayes import NaiveBayesModel
from model.random_forest import RandomForestModel
from model.xgboost_model import XGBoostModel

class HeartDiseaseApp:
    def __init__(self):
        self.model_classes = {
            'K-Nearest Neighbors': KNNModel,
            'Logistic Regression': LogisticRegressionModel,
            'Decision Tree': DecisionTreeModel,
            'Naive Bayes': NaiveBayesModel,
            'Random Forest': RandomForestModel,
            'XGBoost': XGBoostModel
        }
        
        self.model_files = {
            'K-Nearest Neighbors': 'model/knn_model.pkl',
            'Logistic Regression': 'model/logistic_regression_model.pkl',
            'Decision Tree': 'model/decision_tree_model.pkl',
            'Naive Bayes': 'model/naive_bayes_model.pkl',
            'Random Forest': 'model/random_forest_model.pkl',
            'XGBoost': 'model/xgboost_model.pkl'
        }
        
        # Initialize session state
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = {}
        if 'page' not in st.session_state:
            st.session_state.page = "Model Training"
    
    def load_model(self, model_name):
        """Load a trained model"""
        try:
            model_class = self.model_classes[model_name]
            model = model_class()
            model_file = self.model_files[model_name]
            
            if os.path.exists(model_file):
                model.load_model(model_file)
                return model
            else:
                st.error(f"Model file not found: {model_file}")
                return None
        except Exception as e:
            st.error(f"Error loading model {model_name}: {e}")
            return None
    
    def create_sample_data(self):
        """Create sample data for download"""
        sample_data = {
            'age': [63, 37, 41, 56, 57],
            'sex': [1, 1, 0, 1, 0],
            'cp': [3, 2, 1, 1, 0],
            'trestbps': [145, 130, 130, 120, 120],
            'chol': [233, 250, 204, 236, 354],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [0, 1, 0, 1, 1],
            'thalach': [150, 187, 172, 178, 163],
            'exang': [0, 0, 0, 0, 1],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
            'slope': [0, 0, 2, 2, 2],
            'ca': [0, 0, 0, 0, 0],
            'thal': [1, 2, 2, 2, 2],
            'target': [1, 1, 1, 1, 1]
        }
        return pd.DataFrame(sample_data)
    
    def train_models(self):
        """Train all models"""
        data_path = r"C:\Users\Asus\heart.csv"
        
        if not os.path.exists(data_path):
            st.error("Heart dataset not found. Please ensure the file exists at C:\\Users\\Asus\\heart.csv")
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model_metrics = {}
        
        for i, (model_name, model_class) in enumerate(self.model_classes.items()):
            status_text.text(f'Training {model_name}...')
            
            try:
                model = model_class()
                metrics = model.train(data_path)
                
                if metrics:
                    model_file = self.model_files[model_name]
                    model.save_model(model_file)
                    model_metrics[model_name] = metrics
                    st.success(f"✓ {model_name} trained successfully")
                else:
                    st.error(f"✗ {model_name} training failed")
                    
            except Exception as e:
                st.error(f"✗ {model_name} training failed: {e}")
            
            progress_bar.progress((i + 1) / len(self.model_classes))
        
        st.session_state.model_metrics = model_metrics
        st.session_state.models_trained = True
        status_text.text('All models trained!')
        
        return True
    
    def display_metrics_table(self):
        """Display model performance metrics as a table"""
        if not st.session_state.model_metrics:
            st.warning("No metrics available. Please train models first.")
            return None
        
        # Create DataFrame from metrics
        metrics_df = pd.DataFrame(st.session_state.model_metrics).T
        metrics_df = metrics_df.round(4)
        
        return metrics_df
    
    def plot_model_comparison_bar(self, metrics_df):
        """Create bar chart comparing all models across all metrics"""
        fig = go.Figure()
        
        metrics = ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score', 'mcc_score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metrics_df.index,
                y=metrics_df[metric],
                marker_color=colors[i],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Model Performance Comparison - All Metrics",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template="plotly_white"
        )
        
        return fig
    
    def plot_heatmap(self, metrics_df):
        """Create a heatmap of model performance"""
        fig = go.Figure(data=go.Heatmap(
            z=metrics_df.T.values,
            x=metrics_df.index,
            y=metrics_df.columns,
            colorscale='RdYlGn',
            text=metrics_df.T.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Model Performance Heatmap",
            xaxis_title="Models",
            yaxis_title="Metrics",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def plot_radar_chart(self, metrics_df):
        """Create radar chart for top 3 models"""
        avg_performance = metrics_df.mean(axis=1)
        top_3_models = avg_performance.nlargest(3).index
        
        metrics = ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score', 'mcc_score']
        
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, model in enumerate(top_3_models):
            fig.add_trace(go.Scatterpolar(
                r=metrics_df.loc[model, metrics].values,
                theta=[m.replace('_', ' ').title() for m in metrics],
                fill='toself',
                name=model,
                line_color=colors[i],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Top 3 Models - Radar Chart",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def plot_model_ranking(self, metrics_df):
        """Create a bar chart showing model ranking by average performance"""
        avg_performance = metrics_df.mean(axis=1).sort_values(ascending=True)
        
        fig = go.Figure(data=go.Bar(
            x=avg_performance.values,
            y=avg_performance.index,
            orientation='h',
            marker=dict(
                color=avg_performance.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=avg_performance.values.round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Model Ranking by Average Performance",
            xaxis_title="Average Score",
            yaxis_title="Models",
            height=400,
            template="plotly_white",
            xaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_metric_distribution(self, metrics_df):
        """Create box plots showing distribution of metrics across models"""
        df_melted = metrics_df.melt(var_name='Metric', value_name='Score', ignore_index=False)
        df_melted = df_melted.reset_index().rename(columns={'index': 'Model'})
        
        fig = px.box(df_melted, x='Metric', y='Score', color='Metric',
                     title="Distribution of Scores by Metric",
                     points="all",
                     hover_data=['Model'])
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Metrics",
            yaxis_title="Score"
        )
        
        return fig
    
    def display_training_page(self):
        """Display Model Training page with graphs"""
        st.header("📊 Model Training")
        
        st.info("Train all 6 machine learning models on the heart disease dataset")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Train All Models", type="primary", use_container_width=True):
                with st.spinner("Training models... This may take a few minutes."):
                    if self.train_models():
                        st.success("All models trained successfully!")
                        st.rerun()
        
        with col2:
            if st.session_state.models_trained:
                st.success("✅ Models are already trained!")
        
        st.markdown("---")
        
        if st.session_state.models_trained and st.session_state.model_metrics:
            metrics_df = self.display_metrics_table()
            
            if metrics_df is not None:
                # Display metrics table
                st.subheader("📋 Model Performance Metrics Table")
                st.dataframe(metrics_df, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📈 Performance Visualizations")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Bar Chart", 
                    "🔥 Heatmap", 
                    "🎯 Radar Chart",
                    "🏆 Model Ranking",
                    "📦 Distribution"
                ])
                
                with tab1:
                    st.plotly_chart(self.plot_model_comparison_bar(metrics_df), use_container_width=True)
                    
                    # Add insights
                    best_model = metrics_df.mean(axis=1).idxmax()
                    st.info(f"💡 **Insight:** The grouped bar chart shows **{best_model}** performing well across most metrics. Higher bars indicate better performance.")
                
                with tab2:
                    st.plotly_chart(self.plot_heatmap(metrics_df), use_container_width=True)
                    
                    # Find best and worst
                    best_model = metrics_df.mean(axis=1).idxmax()
                    best_metric = metrics_df.mean().idxmax()
                    worst_metric = metrics_df.mean().idxmin()
                    
                    st.info(f"💡 **Insights:**"
                           f"\n- 🟢 **Best Overall Model:** {best_model}"
                           f"\n- 📈 **Strongest Metric:** {best_metric.replace('_', ' ').title()}"
                           f"\n- 📉 **Weakest Metric:** {worst_metric.replace('_', ' ').title()}")
                
                with tab3:
                    st.plotly_chart(self.plot_radar_chart(metrics_df), use_container_width=True)
                    st.info("💡 **Insight:** The radar chart shows the performance profile of the top 3 models. Models with larger, more balanced areas are better rounded.")
                
                with tab4:
                    st.plotly_chart(self.plot_model_ranking(metrics_df), use_container_width=True)
                    
                    # Show ranking list
                    avg_performance = metrics_df.mean(axis=1).sort_values(ascending=False)
                    st.write("**Detailed Ranking:**")
                    for i, (model, score) in enumerate(avg_performance.items(), 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                        st.write(f"{medal} **{model}**: {score:.4f}")
                
                with tab5:
                    st.plotly_chart(self.plot_metric_distribution(metrics_df), use_container_width=True)
                    
                    # Show statistics
                    st.write("**Metric Statistics:**")
                    stats_df = metrics_df.describe().loc[['mean', 'std', 'min', 'max']].round(4)
                    st.dataframe(stats_df, use_container_width=True)
                
                st.markdown("---")
                
                # Best performing models summary
                st.subheader("🏆 Best Models by Individual Metric")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_accuracy = metrics_df['accuracy'].idxmax()
                    best_acc_score = metrics_df['accuracy'].max()
                    st.metric("Best Accuracy", best_accuracy, f"{best_acc_score:.4f}")
                    
                    best_auc = metrics_df['auc_score'].idxmax()
                    best_auc_score = metrics_df['auc_score'].max()
                    st.metric("Best AUC Score", best_auc, f"{best_auc_score:.4f}")
                
                with col2:
                    best_precision = metrics_df['precision'].idxmax()
                    best_prec_score = metrics_df['precision'].max()
                    st.metric("Best Precision", best_precision, f"{best_prec_score:.4f}")
                    
                    best_recall = metrics_df['recall'].idxmax()
                    best_rec_score = metrics_df['recall'].max()
                    st.metric("Best Recall", best_recall, f"{best_rec_score:.4f}")
                
                with col3:
                    best_f1 = metrics_df['f1_score'].idxmax()
                    best_f1_score = metrics_df['f1_score'].max()
                    st.metric("Best F1 Score", best_f1, f"{best_f1_score:.4f}")
                    
                    best_mcc = metrics_df['mcc_score'].idxmax()
                    best_mcc_score = metrics_df['mcc_score'].max()
                    st.metric("Best MCC Score", best_mcc, f"{best_mcc_score:.4f}")
    
    def display_prediction_page(self):
        """Display Prediction page"""
        st.header("🔮 Heart Disease Prediction")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Prediction",
            list(self.model_classes.keys())
        )
        
        # Data input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload Excel File", "Manual Input"]
        )
        
        if input_method == "Upload Excel File":
            self.display_batch_prediction(selected_model)
        else:
            self.display_manual_prediction(selected_model)
    
    def display_batch_prediction(self, selected_model):
        """Display batch prediction interface"""
        st.subheader("Upload Test Data")
        
        # Download sample template
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Sample Template"):
                sample_df = self.create_sample_data()
                template_df = sample_df.drop('target', axis=1)
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    template_df.to_excel(writer, sheet_name='Heart_Data', index=False)
                
                st.download_button(
                    label="Download Excel Template",
                    data=output.getvalue(),
                    file_name="heart_disease_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload Excel file",
                type=['xlsx', 'xls'],
                help="Upload Excel file with patient data (without target column)"
            )
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                
                st.subheader("Uploaded Data Preview")
                st.dataframe(df)
                
                if st.button("🔍 Make Predictions", type="primary"):
                    model = self.load_model(selected_model)
                    
                    if model is not None:
                        try:
                            predictions = model.predict(df)
                            probabilities = model.predict_proba(df)
                            
                            results_df = df.copy()
                            results_df['Prediction'] = predictions
                            results_df['Risk_Probability'] = probabilities[:, 1]
                            results_df['Risk_Level'] = ['High Risk' if p > 0.5 else 'Low Risk' 
                                                      for p in probabilities[:, 1]]
                            
                            st.subheader("Prediction Results")
                            st.dataframe(results_df)
                            
                            # Summary statistics
                            high_risk_count = sum(predictions)
                            total_patients = len(predictions)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Patients", total_patients)
                            with col2:
                                st.metric("High Risk", high_risk_count)
                            with col3:
                                st.metric("Low Risk", total_patients - high_risk_count)
                            
                            # Visualize prediction distribution
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=("Prediction Distribution", "Risk Probability Distribution"),
                                specs=[[{"type": "pie"}, {"type": "histogram"}]]
                            )
                            
                            fig.add_trace(
                                go.Pie(labels=['Low Risk', 'High Risk'], 
                                       values=[total_patients - high_risk_count, high_risk_count],
                                       marker_colors=['green', 'red']),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Histogram(x=probabilities[:, 1], 
                                            nbinsx=20,
                                            marker_color='lightblue',
                                            name='Risk Probability'),
                                row=1, col=2
                            )
                            
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                results_df.to_excel(writer, sheet_name='Results', index=False)
                            
                            st.download_button(
                                label="📥 Download Results",
                                data=output.getvalue(),
                                file_name=f"heart_disease_predictions_{selected_model.lower().replace(' ', '_')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {e}")
                    else:
                        st.error("Could not load the selected model. Please train models first.")
            
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
    
    def display_manual_prediction(self, selected_model):
        """Display manual prediction interface"""
        st.subheader("Enter Patient Data Manually")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120)
            chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise", [0, 1, 2])
            ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
        
        if st.button("🔍 Predict", type="primary"):
            input_data = pd.DataFrame({
                'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
                'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
                'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
            })
            
            model = self.load_model(selected_model)
            
            if model is not None:
                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    
                    st.subheader("Prediction Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error(f"⚠️ **HIGH RISK** of heart disease")
                            st.error(f"Risk Probability: {probability[1]:.3f}")
                        else:
                            st.success(f"✅ **LOW RISK** of heart disease")
                            st.success(f"Risk Probability: {probability[1]:.3f}")
                    
                    with col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability[1] * 100,
                            title={'text': "Risk Probability (%)"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgreen"},
                                    {'range': [50, 100], 'color': "lightcoral"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50}
                            }
                        ))
                        
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.error("Could not load the selected model. Please train models first.")
    
    def display_comparison_page(self):
        """Display Model Comparison page"""
        st.header("📊 Model Performance Comparison")
        
        if st.session_state.models_trained and st.session_state.model_metrics:
            metrics_df = self.display_metrics_table()
            
            if metrics_df is not None:
                # Display table
                st.subheader("📋 Performance Metrics Table")
                st.dataframe(metrics_df, use_container_width=True)
                
                st.markdown("---")
                
                # Create two columns for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Model Ranking")
                    avg_performance = metrics_df.mean(axis=1).sort_values(ascending=False)
                    for i, (model, score) in enumerate(avg_performance.items(), 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                        st.write(f"{medal} **{model}**: {score:.4f}")
                    
                    st.markdown("---")
                    best_overall = avg_performance.index[0]
                    st.success(f"🏆 **Recommended Model**: {best_overall}")
                
                with col2:
                    st.subheader("📊 Best by Metric")
                    for metric in metrics_df.columns:
                        best = metrics_df[metric].idxmax()
                        score = metrics_df[metric].max()
                        st.write(f"- **{metric.replace('_', ' ').title()}**: {best} ({score:.4f})")
                
                st.markdown("---")
                
                # Comparison visualizations
                st.subheader("📈 Comparison Visualizations")
                
                viz_option = st.selectbox(
                    "Select Visualization Type",
                    ["Bar Chart", "Heatmap", "Radar Chart", "Distribution"]
                )
                
                if viz_option == "Bar Chart":
                    st.plotly_chart(self.plot_model_comparison_bar(metrics_df), use_container_width=True)
                elif viz_option == "Heatmap":
                    st.plotly_chart(self.plot_heatmap(metrics_df), use_container_width=True)
                elif viz_option == "Radar Chart":
                    st.plotly_chart(self.plot_radar_chart(metrics_df), use_container_width=True)
                else:
                    st.plotly_chart(self.plot_metric_distribution(metrics_df), use_container_width=True)
        else:
            st.warning("⚠️ Please train the models first to see the comparison.")
            st.info("Go to **Model Training** page to train all models.")
    
    def run(self):
        """Main application interface"""
        st.set_page_config(
            page_title="Heart Disease Prediction",
            page_icon="❤️",
            layout="wide"
        )
        
        st.title("❤️ Heart Disease Prediction System")
        st.markdown("---")
        
        # Sidebar navigation with 3 separate pages (no dropdown)
        #st.sidebar.title("Navigation")
        
        # Create three columns in sidebar for navigation buttons
        #st.sidebar.markdown("### 📍 Pages")
        
        # Button for Model Training
        if st.sidebar.button("📊 Train", use_container_width=True):
            st.session_state.page = "Model Training"

        # Button for Prediction
        if st.sidebar.button("🔮 Predict", use_container_width=True):
            st.session_state.page = "Prediction"

        # Button for Model Comparison
        if st.sidebar.button("📈 Compare", use_container_width=True):
            st.session_state.page = "Model Comparison"  
        # Show current page indicator
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Current Page:** {st.session_state.page}")
        
        # Display training status in sidebar
        if st.session_state.models_trained:
            st.sidebar.success("✅ Models Trained")
        else:
            st.sidebar.warning("⚠️ Models Not Trained")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About")
        st.sidebar.info(
            "This app uses 6 machine learning models to predict heart disease risk "
            "based on patient medical data."
        )
        
        # Display the selected page
        if st.session_state.page == "Model Training":
            self.display_training_page()
        elif st.session_state.page == "Prediction":
            self.display_prediction_page()
        else:  # Model Comparison
            self.display_comparison_page()

def main():
    app = HeartDiseaseApp()
    app.run()

if __name__ == "__main__":
    main()