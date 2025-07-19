import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train logistic regression model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training Logistic Regression...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store model and scaler
        self.models['logistic_regression'] = {
            'model': model,
            'scaler': scaler,
            'train_score': train_score,
            'test_score': test_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Logistic Regression - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return self.models['logistic_regression']
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train random forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Training Random Forest...")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        self.feature_importance['random_forest'] = feature_importance
        
        # Store model
        self.models['random_forest'] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
        
        logger.info(f"Random Forest - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return self.models['random_forest']
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model."""
        import xgboost as xgb
        
        logger.info("Training XGBoost...")
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        self.feature_importance['xgboost'] = feature_importance
        
        # Store model
        self.models['xgboost'] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
        
        logger.info(f"XGBoost - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return self.models['xgboost']
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train LightGBM model."""
        import lightgbm as lgb
        
        logger.info("Training LightGBM...")
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        self.feature_importance['lightgbm'] = feature_importance
        
        # Store model
        self.models['lightgbm'] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
        
        logger.info(f"LightGBM - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return self.models['lightgbm']
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Train all models and return results."""
        logger.info("Training all models...")
        
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_lightgbm(X_train, y_train, X_test, y_test)
        
        # Select best model based on test score
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['test_score'])
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with test score: {self.best_model['test_score']:.4f}")
        
        return self.models
    
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate model performance."""
        logger.info(f"Evaluating {model_name}...")
        
        # Classification metrics
        accuracy = (y_true == y_pred).mean()
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        evaluation_results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        return evaluation_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str = 'xgboost') -> Dict[str, Any]:
        """Perform hyperparameter tuning for the specified model."""
        logger.info(f"Hyperparameter tuning for {model_name}...")
        
        if model_name == 'xgboost':
            import xgboost as xgb
            
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def save_model(self, model_name: str, file_path: str) -> None:
        """Save trained model to file."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], file_path)
        logger.info(f"Model {model_name} saved to {file_path}")
    
    def load_model(self, file_path: str) -> Dict[str, Any]:
        """Load trained model from file."""
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model_data
    
    def predict_no_show_probability(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Predict no-show probability for new data."""
        if model_name is None:
            model_name = 'best'
            model_data = self.best_model
        else:
            model_data = self.models[model_name]
        
        model = model_data['model']
        
        # Apply scaling if needed
        if 'scaler' in model_data:
            X_scaled = model_data['scaler'].transform(X)
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if model_name is None:
            model_name = list(self.feature_importance.keys())[0]
        
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        importance = self.feature_importance[model_name]
        
        # Sort by importance and return top N
        sorted_importance = dict(sorted(importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:top_n])
        
        return sorted_importance