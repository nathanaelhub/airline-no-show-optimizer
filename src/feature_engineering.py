import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from booking and departure dates."""
        logger.info("Creating temporal features...")
        
        # Days between booking and departure
        df['days_to_departure'] = (df['departure_date'] - df['booking_date']).dt.days
        
        # Booking timing features
        df['booking_hour'] = df['booking_date'].dt.hour
        df['booking_day_of_week'] = df['booking_date'].dt.dayofweek
        df['booking_month'] = df['booking_date'].dt.month
        df['booking_quarter'] = df['booking_date'].dt.quarter
        
        # Departure timing features
        df['departure_hour'] = df['departure_date'].dt.hour
        df['departure_day_of_week'] = df['departure_date'].dt.dayofweek
        df['departure_month'] = df['departure_date'].dt.month
        df['departure_quarter'] = df['departure_date'].dt.quarter
        
        # Weekend indicators
        df['is_weekend_booking'] = (df['booking_day_of_week'].isin([5, 6])).astype(int)
        df['is_weekend_departure'] = (df['departure_day_of_week'].isin([5, 6])).astype(int)
        
        # Holiday indicators (simplified)
        df['is_holiday_season'] = df['departure_month'].isin([12, 1, 7, 8]).astype(int)
        
        # Early morning/late night flights
        df['is_early_morning'] = (df['departure_hour'] < 6).astype(int)
        df['is_late_night'] = (df['departure_hour'] >= 22).astype(int)
        
        # Advance booking categories
        df['booking_category'] = pd.cut(
            df['days_to_departure'], 
            bins=[-1, 1, 7, 21, 60, float('inf')],
            labels=['same_day', 'week_before', 'three_weeks', 'two_months', 'advance']
        )
        
        logger.info("Temporal features created")
        return df
    
    def create_passenger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create passenger-specific features."""
        logger.info("Creating passenger features...")
        
        # Frequent flyer indicators
        passenger_stats = df.groupby('passenger_id').agg({
            'booking_date': 'count',
            'no_show': 'sum',
            'ticket_price': 'mean'
        }).rename(columns={
            'booking_date': 'total_bookings',
            'no_show': 'total_no_shows',
            'ticket_price': 'avg_ticket_price'
        })
        
        # Calculate no-show rate
        passenger_stats['no_show_rate'] = (
            passenger_stats['total_no_shows'] / passenger_stats['total_bookings']
        ).fillna(0)
        
        # Passenger loyalty tier
        passenger_stats['loyalty_tier'] = pd.cut(
            passenger_stats['total_bookings'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['new', 'occasional', 'regular', 'frequent']
        )
        
        # Merge back to main dataset
        df = df.merge(passenger_stats, on='passenger_id', how='left')
        
        logger.info("Passenger features created")
        return df
    
    def create_flight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create flight-specific features."""
        logger.info("Creating flight features...")
        
        # Flight route popularity
        route_stats = df.groupby(['origin', 'destination']).agg({
            'flight_id': 'count',
            'no_show': 'mean'
        }).rename(columns={
            'flight_id': 'route_frequency',
            'no_show': 'route_no_show_rate'
        })
        
        df = df.merge(route_stats, on=['origin', 'destination'], how='left')
        
        # Flight duration categories
        df['flight_duration_category'] = pd.cut(
            df['flight_duration'],
            bins=[0, 2, 4, 8, float('inf')],
            labels=['short', 'medium', 'long', 'ultra_long']
        )
        
        # Aircraft type indicators
        df['is_wide_body'] = df['aircraft_type'].str.contains('777|787|A330|A340|A350|A380', na=False).astype(int)
        
        # Seat class indicators
        df['is_premium_class'] = df['seat_class'].isin(['business', 'first']).astype(int)
        
        logger.info("Flight features created")
        return df
    
    def create_pricing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pricing-related features."""
        logger.info("Creating pricing features...")
        
        # Price percentiles by route
        route_price_stats = df.groupby(['origin', 'destination'])['ticket_price'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).add_prefix('route_price_')
        
        df = df.merge(route_price_stats, on=['origin', 'destination'], how='left')
        
        # Price relative to route average
        df['price_vs_route_avg'] = df['ticket_price'] / df['route_price_mean']
        df['price_vs_route_median'] = df['ticket_price'] / df['route_price_median']
        
        # Price categories
        df['price_category'] = pd.cut(
            df['ticket_price'],
            bins=[0, 200, 500, 1000, float('inf')],
            labels=['budget', 'economy', 'premium', 'luxury']
        )
        
        # Discount indicators
        df['is_discounted'] = (df['price_vs_route_avg'] < 0.8).astype(int)
        df['is_premium_priced'] = (df['price_vs_route_avg'] > 1.2).astype(int)
        
        logger.info("Pricing features created")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different categories."""
        logger.info("Creating interaction features...")
        
        # Passenger type + booking timing
        df['frequent_last_minute'] = (
            (df['loyalty_tier'] == 'frequent') & 
            (df['booking_category'] == 'same_day')
        ).astype(int)
        
        # Price sensitivity + advance booking
        df['price_sensitive_advance'] = (
            (df['price_category'] == 'budget') & 
            (df['booking_category'] == 'advance')
        ).astype(int)
        
        # Weekend + holiday combination
        df['weekend_holiday'] = (
            df['is_weekend_departure'] & 
            df['is_holiday_season']
        ).astype(int)
        
        logger.info("Interaction features created")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        # One-hot encode low cardinality categoricals
        categorical_cols = ['seat_class', 'aircraft_type', 'booking_category', 
                          'flight_duration_category', 'price_category', 'loyalty_tier']
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # Label encode high cardinality categoricals
        from sklearn.preprocessing import LabelEncoder
        
        high_cardinality_cols = ['origin', 'destination', 'airline']
        for col in high_cardinality_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        logger.info("Categorical encoding completed")
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features in the correct order."""
        logger.info("Starting feature engineering pipeline...")
        
        df = self.create_temporal_features(df)
        df = self.create_passenger_features(df)
        df = self.create_flight_features(df)
        df = self.create_pricing_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df)
        
        logger.info("Feature engineering completed")
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       feature_importance_threshold: float = 0.01) -> List[str]:
        """Select most important features using feature importance."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        
        logger.info("Selecting important features...")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-numeric columns for feature selection
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)
        
        # Select features based on importance
        selector = SelectFromModel(rf, threshold=feature_importance_threshold)
        selector.fit(X_numeric, y)
        
        selected_features = numeric_cols[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} important features")
        return selected_features