import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirlineRevenueMetrics:
    """
    Revenue-focused evaluation metrics for airline no-show prediction.
    
    Key Business Metrics:
    - Cost of denied boarding: $800-1200 per passenger
    - Cost of empty seat: $200-400 per seat (opportunity cost)
    - Revenue per passenger: Varies by route/class
    - Overbooking optimization: Balance between denied boarding and empty seats
    """
    
    def __init__(self, denied_boarding_cost: float = 1000, 
                 empty_seat_cost: float = 300,
                 average_ticket_price: float = 400):
        self.denied_boarding_cost = denied_boarding_cost
        self.empty_seat_cost = empty_seat_cost
        self.average_ticket_price = average_ticket_price
        
    def calculate_revenue_impact(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray, ticket_prices: np.ndarray = None) -> Dict[str, float]:
        """Calculate revenue impact of predictions vs actual outcomes."""
        
        if ticket_prices is None:
            ticket_prices = np.full(len(y_true), self.average_ticket_price)
        
        # Confusion matrix components
        tn = np.sum((y_true == 0) & (y_pred == 0))  # Correctly predicted show-ups
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False no-show predictions
        fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed no-shows
        tp = np.sum((y_true == 1) & (y_pred == 1))  # Correctly predicted no-shows
        
        # Revenue impact calculations
        # FN (missed no-shows) = opportunity cost of empty seats
        missed_no_show_cost = fn * self.empty_seat_cost
        
        # FP (false no-show predictions) = potential overbooking if acted upon
        false_alarm_cost = fp * (self.denied_boarding_cost * 0.1)  # 10% chance of acting on prediction
        
        # TP (correct no-show predictions) = revenue saved through proper planning
        correct_prediction_value = tp * self.empty_seat_cost * 0.8  # 80% recovery through overbooking
        
        # Total cost
        total_cost = missed_no_show_cost + false_alarm_cost
        total_benefit = correct_prediction_value
        net_revenue_impact = total_benefit - total_cost
        
        # Calculate metrics
        revenue_metrics = {
            'missed_no_show_cost': missed_no_show_cost,
            'false_alarm_cost': false_alarm_cost,
            'correct_prediction_value': correct_prediction_value,
            'net_revenue_impact': net_revenue_impact,
            'cost_per_passenger': total_cost / len(y_true),
            'revenue_improvement_per_passenger': net_revenue_impact / len(y_true),
            'total_ticket_revenue': np.sum(ticket_prices),
            'revenue_impact_percentage': (net_revenue_impact / np.sum(ticket_prices)) * 100
        }
        
        return revenue_metrics
    
    def calculate_overbooking_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    capacity: int = 180, current_bookings: int = 200) -> Dict[str, float]:
        """Calculate overbooking-specific metrics."""
        
        # Simulate overbooking scenarios based on predictions
        predicted_no_shows = np.sum(y_pred_proba > 0.5)
        actual_no_shows = np.sum(y_true)
        
        # Calculate optimal overbooking based on predictions
        predicted_passengers = current_bookings - predicted_no_shows
        actual_passengers = current_bookings - actual_no_shows
        
        # Denied boarding scenarios
        predicted_denied = max(0, predicted_passengers - capacity)
        actual_denied = max(0, actual_passengers - capacity)
        
        # Empty seat scenarios  
        predicted_empty = max(0, capacity - predicted_passengers)
        actual_empty = max(0, capacity - actual_passengers)
        
        overbooking_metrics = {
            'predicted_passengers': predicted_passengers,
            'actual_passengers': actual_passengers,
            'passenger_prediction_error': abs(predicted_passengers - actual_passengers),
            'predicted_denied_boarding': predicted_denied,
            'actual_denied_boarding': actual_denied,
            'predicted_empty_seats': predicted_empty,
            'actual_empty_seats': actual_empty,
            'load_factor_predicted': predicted_passengers / capacity,
            'load_factor_actual': actual_passengers / capacity,
            'overbooking_accuracy': 1 - (abs(predicted_passengers - actual_passengers) / capacity)
        }
        
        return overbooking_metrics


class CostSensitiveLearning:
    """
    Implement cost-sensitive learning for airline no-show prediction.
    
    Business Context:
    - False Negative (missing no-show) = empty seat cost
    - False Positive (false alarm) = potential denied boarding cost
    - Asymmetric costs require weighted learning
    """
    
    def __init__(self, fn_cost: float = 300, fp_cost: float = 100):
        self.fn_cost = fn_cost  # Cost of missing a no-show (empty seat)
        self.fp_cost = fp_cost  # Cost of false alarm (potential overbooking issue)
        self.cost_ratio = fn_cost / fp_cost
        
    def calculate_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Calculate sample weights based on cost sensitivity."""
        weights = np.ones(len(y))
        
        # Weight positive class (no-shows) higher due to higher cost of missing them
        weights[y == 1] = self.cost_ratio
        weights[y == 0] = 1.0
        
        return weights
    
    def calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for cost-sensitive learning."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Calculate weights inversely proportional to class frequency, adjusted for costs
        class_weights = {}
        for class_label in np.unique(y):
            class_freq = np.sum(y == class_label) / n_samples
            if class_label == 1:  # No-show class
                class_weights[class_label] = (1 / class_freq) * self.cost_ratio
            else:  # Show-up class
                class_weights[class_label] = 1 / class_freq
                
        return class_weights
    
    def calculate_cost_sensitive_threshold(self, y_true: np.ndarray, 
                                         y_pred_proba: np.ndarray) -> float:
        """Find optimal threshold based on cost-sensitive criteria."""
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_cost = float('inf')
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate costs
            fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed no-shows
            fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
            
            total_cost = (fn * self.fn_cost) + (fp * self.fp_cost)
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
                
        return best_threshold


class ModelComparisonFramework:
    """
    Comprehensive model comparison framework for airline no-show prediction
    with revenue-focused metrics and cost-sensitive learning.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.revenue_metrics = AirlineRevenueMetrics()
        self.cost_sensitive = CostSensitiveLearning()
        
        # Set random seeds
        np.random.seed(random_state)
        
    def create_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Create baseline model using industry standard no-show rates.
        
        Industry Baseline: 5-7% no-show rate applied uniformly
        """
        logger.info("Creating baseline model...")
        
        # Industry standard no-show rate
        industry_no_show_rate = 0.06  # 6% industry average
        
        class BaselineModel:
            def __init__(self, no_show_rate: float = 0.06):
                self.no_show_rate = no_show_rate
                
            def predict_proba(self, X):
                # Return constant probability for all passengers
                n_samples = len(X)
                proba = np.full((n_samples, 2), [1 - self.no_show_rate, self.no_show_rate])
                return proba
            
            def predict(self, X):
                # Predict no-show for top percentage based on industry rate
                n_samples = len(X)
                n_no_shows = int(n_samples * self.no_show_rate)
                predictions = np.zeros(n_samples)
                predictions[:n_no_shows] = 1
                return predictions
        
        baseline_model = BaselineModel(industry_no_show_rate)
        
        self.models['baseline'] = {
            'model': baseline_model,
            'model_type': 'baseline',
            'description': f'Industry standard {industry_no_show_rate:.1%} no-show rate',
            'hyperparameters': {'no_show_rate': industry_no_show_rate}
        }
        
        logger.info("Baseline model created")
        return self.models['baseline']
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train logistic regression with interpretation capabilities."""
        logger.info("Training Logistic Regression...")
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Calculate class weights for cost-sensitive learning
        class_weights = self.cost_sensitive.calculate_class_weights(y_train)
        
        # Train model
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight=class_weights,
            solver='liblinear'
        )
        model.fit(X_train_scaled, y_train)
        
        # Feature interpretation
        feature_coefficients = dict(zip(X_train.columns, model.coef_[0]))
        feature_importance = {k: abs(v) for k, v in feature_coefficients.items()}
        
        self.models['logistic_regression'] = {
            'model': model,
            'scaler': scaler,
            'model_type': 'linear',
            'description': 'Logistic Regression with cost-sensitive weights',
            'feature_coefficients': feature_coefficients,
            'feature_importance': feature_importance,
            'hyperparameters': {'class_weight': class_weights, 'C': 1.0}
        }
        
        logger.info("Logistic Regression trained")
        return self.models['logistic_regression']
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train Random Forest with feature importance analysis."""
        logger.info("Training Random Forest...")
        
        # Calculate sample weights for cost-sensitive learning
        sample_weights = self.cost_sensitive.calculate_sample_weights(y_train)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Feature importance analysis
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Get top features
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:20])
        
        self.models['random_forest'] = {
            'model': model,
            'model_type': 'ensemble',
            'description': 'Random Forest with cost-sensitive sample weights',
            'feature_importance': feature_importance,
            'top_features': top_features,
            'hyperparameters': {
                'n_estimators': 200,
                'max_depth': 15,
                'class_weight': 'balanced'
            }
        }
        
        logger.info("Random Forest trained")
        return self.models['random_forest']
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train XGBoost for optimal performance."""
        logger.info("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        scale_pos_weight *= self.cost_sensitive.cost_ratio  # Adjust for cost sensitivity
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            early_stopping_rounds=50
        )
        
        # Split training data for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            verbose=False
        )
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:20])
        
        self.models['xgboost'] = {
            'model': model,
            'model_type': 'gradient_boosting',
            'description': 'XGBoost with cost-sensitive scale_pos_weight',
            'feature_importance': feature_importance,
            'top_features': top_features,
            'hyperparameters': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'scale_pos_weight': scale_pos_weight
            }
        }
        
        logger.info("XGBoost trained")
        return self.models['xgboost']
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train LightGBM as additional ensemble method."""
        logger.info("Training LightGBM...")
        
        # Calculate class weights
        class_weights = self.cost_sensitive.calculate_class_weights(y_train)
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weights,
            random_state=self.random_state,
            verbose=-1,
            early_stopping_rounds=50
        )
        
        # Split for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:20])
        
        self.models['lightgbm'] = {
            'model': model,
            'model_type': 'gradient_boosting',
            'description': 'LightGBM with cost-sensitive class weights',
            'feature_importance': feature_importance,
            'top_features': top_features,
            'hyperparameters': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'class_weight': class_weights
            }
        }
        
        logger.info("LightGBM trained")
        return self.models['lightgbm']
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series,
                      ticket_prices: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive model evaluation with revenue metrics."""
        logger.info(f"Evaluating {model_name}...")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Make predictions
        if model_name == 'baseline':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            if 'scaler' in model_info:
                X_test_scaled = model_info['scaler'].transform(X_test)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
        
        # Find optimal threshold using cost-sensitive approach
        optimal_threshold = self.cost_sensitive.calculate_cost_sensitive_threshold(
            y_test.values, y_pred_proba
        )
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Standard metrics
        accuracy = np.mean(y_test == y_pred)
        accuracy_optimal = np.mean(y_test == y_pred_optimal)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Revenue metrics
        revenue_metrics = self.revenue_metrics.calculate_revenue_impact(
            y_test.values, y_pred_optimal, y_pred_proba, ticket_prices
        )
        
        # Overbooking metrics
        overbooking_metrics = self.revenue_metrics.calculate_overbooking_metrics(
            y_test.values, y_pred_proba
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel()
        
        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'accuracy_optimal': accuracy_optimal,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'revenue_metrics': revenue_metrics,
            'overbooking_metrics': overbooking_metrics,
            'predictions': y_pred_optimal,
            'probabilities': y_pred_proba
        }
        
        self.results[model_name] = evaluation_results
        
        logger.info(f"{model_name} evaluation completed")
        return evaluation_results
    
    def compare_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          ticket_prices: np.ndarray = None) -> pd.DataFrame:
        """Train and compare all models."""
        logger.info("Starting comprehensive model comparison...")
        
        # Train all models
        self.create_baseline_model(X_train, y_train)
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        # Evaluate all models
        for model_name in self.models.keys():
            self.evaluate_model(model_name, X_test, y_test, ticket_prices)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Accuracy (Optimal)': results['accuracy_optimal'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'AUC Score': results['auc_score'],
                'Revenue Impact ($)': results['revenue_metrics']['net_revenue_impact'],
                'Revenue per Passenger ($)': results['revenue_metrics']['revenue_improvement_per_passenger'],
                'Overbooking Accuracy': results['overbooking_metrics']['overbooking_accuracy'],
                'Optimal Threshold': results['optimal_threshold']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Create visualizations for model comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Accuracy comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['AUC Score'])
        axes[0, 0].set_title('AUC Score Comparison')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Revenue impact
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Revenue Impact ($)'])
        axes[0, 1].set_title('Revenue Impact Comparison')
        axes[0, 1].set_ylabel('Net Revenue Impact ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Precision vs Recall
        axes[0, 2].scatter(comparison_df['Recall'], comparison_df['Precision'], s=100)
        for i, model in enumerate(comparison_df['Model']):
            axes[0, 2].annotate(model, (comparison_df['Recall'].iloc[i], 
                                       comparison_df['Precision'].iloc[i]))
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision vs Recall')
        
        # 4. F1 Score comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['F1 Score'])
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Revenue per passenger
        axes[1, 1].bar(comparison_df['Model'], comparison_df['Revenue per Passenger ($)'])
        axes[1, 1].set_title('Revenue Impact per Passenger')
        axes[1, 1].set_ylabel('Revenue per Passenger ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Overbooking accuracy
        axes[1, 2].bar(comparison_df['Model'], comparison_df['Overbooking Accuracy'])
        axes[1, 2].set_title('Overbooking Prediction Accuracy')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance_comparison(self) -> pd.DataFrame:
        """Compare feature importance across models."""
        feature_importance_data = []
        
        for model_name, model_info in self.models.items():
            if 'feature_importance' in model_info:
                for feature, importance in model_info['feature_importance'].items():
                    feature_importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': importance
                    })
        
        feature_df = pd.DataFrame(feature_importance_data)
        
        # Get top features across all models
        top_features = (feature_df.groupby('Feature')['Importance']
                       .mean()
                       .sort_values(ascending=False)
                       .head(20)
                       .index.tolist())
        
        # Pivot table for comparison
        feature_comparison = feature_df[feature_df['Feature'].isin(top_features)].pivot(
            index='Feature', columns='Model', values='Importance'
        ).fillna(0)
        
        return feature_comparison
    
    def generate_business_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate a business-focused report."""
        best_revenue_model = comparison_df.loc[comparison_df['Revenue Impact ($)'].idxmax()]
        best_auc_model = comparison_df.loc[comparison_df['AUC Score'].idxmax()]
        
        report = f"""
        AIRLINE NO-SHOW PREDICTION - BUSINESS IMPACT REPORT
        ================================================
        
        EXECUTIVE SUMMARY:
        - Best Revenue Model: {best_revenue_model['Model']} (+${best_revenue_model['Revenue Impact ($)']:,.0f} revenue impact)
        - Best Predictive Model: {best_auc_model['Model']} (AUC: {best_auc_model['AUC Score']:.3f})
        - Baseline vs Best: {best_revenue_model['Revenue Impact ($)'] - comparison_df[comparison_df['Model'] == 'baseline']['Revenue Impact ($)'].iloc[0]:+.0f} improvement
        
        KEY FINDINGS:
        1. Revenue Impact: Best model generates ${best_revenue_model['Revenue per Passenger ($)']:.2f} per passenger
        2. Overbooking Accuracy: {best_revenue_model['Overbooking Accuracy']:.1%} passenger count prediction accuracy
        3. Optimal Threshold: {best_revenue_model['Optimal Threshold']:.3f} (vs standard 0.5)
        
        BUSINESS RECOMMENDATIONS:
        1. Deploy {best_revenue_model['Model']} for revenue optimization
        2. Use cost-sensitive thresholds for overbooking decisions
        3. Monitor {best_auc_model['Model']} for pure prediction accuracy
        4. Implement real-time prediction scoring
        
        RISK FACTORS:
        - False negatives cost ${self.cost_sensitive.fn_cost} per missed no-show
        - False positives cost ${self.cost_sensitive.fp_cost} per false alarm
        - Model requires retraining every 3-6 months
        """
        
        return report


def main():
    """Demonstrate the model comparison framework."""
    
    # Use pathlib for cross-platform compatibility
    PROJECT_ROOT = Path(__file__).parent.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'airline_data_enhanced.csv'
    
    # Load enhanced dataset
    df = pd.read_csv(INPUT_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"No-show rate: {df['no_show'].mean():.3%}")
    
    # Prepare features (select numeric features for demo)
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns.remove('no_show')  # Remove target
    
    X = df[feature_columns]
    y = df['no_show']
    ticket_prices = df['ticket_price'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test, prices_train, prices_test = train_test_split(
        X, y, ticket_prices, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize framework
    framework = ModelComparisonFramework(random_state=42)
    
    # Compare all models
    comparison_results = framework.compare_all_models(
        X_train, y_train, X_test, y_test, prices_test
    )
    
    print("\n=== MODEL COMPARISON RESULTS ===")
    print(comparison_results)
    
    # Generate business report
    business_report = framework.generate_business_report(comparison_results)
    print(business_report)
    
    # Feature importance comparison
    feature_importance = framework.get_feature_importance_comparison()
    print("\n=== TOP FEATURE IMPORTANCE ===")
    print(feature_importance.head(10))
    
    # Save results
    RESULTS_DIR = PROJECT_ROOT / 'results'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    comparison_results.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    feature_importance.to_csv(RESULTS_DIR / 'feature_importance.csv')
    
    print(f"\nResults saved to {RESULTS_DIR} directory")


if __name__ == "__main__":
    main()