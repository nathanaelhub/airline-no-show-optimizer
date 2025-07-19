import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from scipy import stats
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for airline no-show prediction with domain-specific insights.
    
    Key Domain Insights:
    - No-show rates increase dramatically for same-day bookings (panic bookings)
    - Early morning flights (before 6 AM) have higher no-show rates due to sleep-in risk
    - Business travelers have different patterns than leisure travelers
    - Weekend departures have different no-show patterns than weekdays
    - Passengers with history of no-shows are more likely to no-show again
    - Expensive routes vs. cheap routes have different no-show behaviors
    - Weather seasons affect no-show patterns significantly
    """
    
    def __init__(self):
        self.feature_columns = []
        self.passenger_history = {}
        self.route_stats = {}
        
    def create_advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time-based features with domain insights.
        
        Domain Insights:
        - Same-day bookings have 3x higher no-show rates
        - 6 AM flights have highest no-show rates (oversleeping)
        - Monday morning flights have higher no-show rates (weekend hangover effect)
        - Holiday travel has lower no-show rates (important trips)
        """
        logger.info("Creating advanced temporal features...")
        
        # Ensure datetime columns
        df['booking_date'] = pd.to_datetime(df['booking_date'])
        df['departure_date'] = pd.to_datetime(df['departure_date'])
        
        # Core temporal features
        df['days_to_departure'] = (df['departure_date'] - df['booking_date']).dt.days
        df['hours_to_departure'] = (df['departure_date'] - df['booking_date']).dt.total_seconds() / 3600
        
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
        
        # Advanced booking urgency features (HIGH PREDICTIVE POWER)
        df['is_same_day_booking'] = (df['days_to_departure'] == 0).astype(int)
        df['is_last_minute'] = (df['days_to_departure'] <= 1).astype(int)
        df['is_very_last_minute'] = (df['hours_to_departure'] <= 6).astype(int)
        df['is_panic_booking'] = (df['hours_to_departure'] <= 2).astype(int)
        
        # Sleep-in risk factors (CRITICAL for early flights)
        df['is_red_eye_risk'] = (df['departure_hour'] < 6).astype(int)
        df['is_early_morning_risk'] = (df['departure_hour'] < 8).astype(int)
        df['sleep_in_risk_score'] = np.where(
            df['departure_hour'] < 6, 3,
            np.where(df['departure_hour'] < 8, 2,
                    np.where(df['departure_hour'] < 10, 1, 0))
        )
        
        # Weekend effect patterns
        df['is_weekend_departure'] = (df['departure_day_of_week'].isin([5, 6])).astype(int)
        df['is_monday_morning'] = ((df['departure_day_of_week'] == 0) & 
                                  (df['departure_hour'] < 12)).astype(int)
        df['is_friday_evening'] = ((df['departure_day_of_week'] == 4) & 
                                  (df['departure_hour'] >= 17)).astype(int)
        
        # Holiday and seasonal patterns
        df['is_major_holiday'] = df['departure_month'].isin([12, 1]).astype(int)
        df['is_summer_peak'] = df['departure_month'].isin([7, 8]).astype(int)
        df['is_thanksgiving_week'] = ((df['departure_month'] == 11) & 
                                     (df['departure_date'].dt.day >= 20)).astype(int)
        df['is_christmas_week'] = ((df['departure_month'] == 12) & 
                                  (df['departure_date'].dt.day >= 20)).astype(int)
        
        # Business vs leisure travel timing patterns
        df['is_business_hours_booking'] = ((df['booking_hour'] >= 9) & 
                                          (df['booking_hour'] <= 17)).astype(int)
        df['is_business_travel_pattern'] = ((df['departure_day_of_week'] < 5) & 
                                           (df['is_business_hours_booking'] == 1)).astype(int)
        
        # Weather season risk (winter = higher no-show risk)
        df['is_winter_weather'] = df['departure_month'].isin([12, 1, 2, 3]).astype(int)
        df['is_storm_season'] = df['departure_month'].isin([6, 7, 8, 9]).astype(int)
        
        # Advance booking categories with domain insights
        df['booking_urgency'] = pd.cut(
            df['days_to_departure'],
            bins=[-1, 0, 1, 3, 7, 21, 60, float('inf')],
            labels=['same_day', 'next_day', 'panic_zone', 'last_week', 
                   'planned', 'well_planned', 'far_advance']
        )
        
        # Time between booking and departure percentiles
        df['booking_timing_percentile'] = df['days_to_departure'].rank(pct=True)
        df['is_extremely_late_booking'] = (df['booking_timing_percentile'] <= 0.05).astype(int)
        df['is_extremely_early_booking'] = (df['booking_timing_percentile'] >= 0.95).astype(int)
        
        logger.info("Advanced temporal features created")
        return df
    
    def create_passenger_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create passenger behavior features based on historical patterns.
        
        Domain Insights:
        - Passengers with previous no-shows are 2-3x more likely to no-show again
        - Frequent flyers have lower no-show rates (brand loyalty)
        - New passengers have higher no-show rates (uncertainty)
        - Passengers who book multiple segments have different patterns
        """
        logger.info("Creating passenger behavior features...")
        
        # Sort by passenger and booking date for historical analysis
        df = df.sort_values(['passenger_id', 'booking_date'])
        
        # Historical booking patterns
        passenger_stats = df.groupby('passenger_id').agg({
            'booking_date': ['count', 'min', 'max'],
            'days_to_departure': ['mean', 'std', 'min', 'max'],
            'ticket_price': ['mean', 'std', 'min', 'max'],
            'no_show': ['sum', 'mean']
        }).round(3)
        
        # Flatten column names
        passenger_stats.columns = ['_'.join(col).strip() for col in passenger_stats.columns]
        
        # Rename for clarity
        passenger_stats = passenger_stats.rename(columns={
            'booking_date_count': 'total_bookings',
            'booking_date_min': 'first_booking_date',
            'booking_date_max': 'last_booking_date',
            'days_to_departure_mean': 'avg_advance_booking',
            'days_to_departure_std': 'booking_consistency',
            'ticket_price_mean': 'avg_ticket_price',
            'ticket_price_std': 'price_variance',
            'no_show_sum': 'total_no_shows',
            'no_show_mean': 'historical_no_show_rate'
        })
        
        # Calculate customer tenure
        passenger_stats['customer_tenure_days'] = (
            passenger_stats['last_booking_date'] - passenger_stats['first_booking_date']
        ).dt.days
        
        # Passenger loyalty indicators
        passenger_stats['is_frequent_flyer'] = (passenger_stats['total_bookings'] >= 5).astype(int)
        passenger_stats['is_very_frequent_flyer'] = (passenger_stats['total_bookings'] >= 10).astype(int)
        passenger_stats['is_new_passenger'] = (passenger_stats['total_bookings'] == 1).astype(int)
        
        # No-show risk categories
        passenger_stats['no_show_risk_category'] = pd.cut(
            passenger_stats['historical_no_show_rate'],
            bins=[0, 0.1, 0.2, 0.3, 1.0],
            labels=['low_risk', 'medium_risk', 'high_risk', 'very_high_risk']
        )
        
        # Booking behavior patterns
        passenger_stats['is_consistent_booker'] = (
            passenger_stats['booking_consistency'] <= passenger_stats['booking_consistency'].quantile(0.25)
        ).astype(int)
        
        passenger_stats['is_price_sensitive'] = (
            passenger_stats['price_variance'] <= passenger_stats['price_variance'].quantile(0.25)
        ).astype(int)
        
        # Reset index to merge back
        passenger_stats = passenger_stats.reset_index()
        
        # Merge back to main dataframe
        df = df.merge(passenger_stats, on='passenger_id', how='left')
        
        # Fill NaN values for new passengers
        df['historical_no_show_rate'] = df['historical_no_show_rate'].fillna(0)
        df['booking_consistency'] = df['booking_consistency'].fillna(0)
        df['price_variance'] = df['price_variance'].fillna(0)
        
        # Additional per-booking features
        df['is_repeat_customer'] = (df['total_bookings'] > 1).astype(int)
        df['bookings_this_year'] = df.groupby('passenger_id')['booking_date'].transform(
            lambda x: x.dt.year.eq(x.dt.year.mode()[0]).sum()
        )
        
        # Recency features
        df['days_since_last_booking'] = (
            df['booking_date'] - df.groupby('passenger_id')['booking_date'].transform('max')
        ).dt.days
        df['days_since_last_booking'] = df['days_since_last_booking'].fillna(0)
        
        logger.info("Passenger behavior features created")
        return df
    
    def create_flight_characteristics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create flight-specific features based on route and operational characteristics.
        
        Domain Insights:
        - Popular routes have lower no-show rates (established patterns)
        - Red-eye flights have higher no-show rates
        - Hub-to-hub routes vs. spoke routes have different patterns
        - Flight duration affects no-show probability
        """
        logger.info("Creating flight characteristics features...")
        
        # Route popularity and performance metrics
        route_stats = df.groupby(['origin', 'destination']).agg({
            'passenger_id': 'count',
            'no_show': ['sum', 'mean'],
            'ticket_price': ['mean', 'std'],
            'days_to_departure': 'mean',
            'flight_duration': 'mean'
        }).round(3)
        
        # Flatten column names
        route_stats.columns = ['_'.join(col).strip() for col in route_stats.columns]
        route_stats = route_stats.rename(columns={
            'passenger_id_count': 'route_volume',
            'no_show_sum': 'route_total_no_shows',
            'no_show_mean': 'route_no_show_rate',
            'ticket_price_mean': 'route_avg_price',
            'ticket_price_std': 'route_price_volatility',
            'days_to_departure_mean': 'route_avg_advance_booking',
            'flight_duration_mean': 'route_avg_duration'
        })
        
        # Route popularity categories
        route_stats['route_popularity'] = pd.cut(
            route_stats['route_volume'],
            bins=[0, 10, 50, 100, float('inf')],
            labels=['rare', 'uncommon', 'popular', 'very_popular']
        )
        
        # Route performance categories
        route_stats['route_reliability'] = pd.cut(
            route_stats['route_no_show_rate'],
            bins=[0, 0.05, 0.1, 0.15, 1.0],
            labels=['very_reliable', 'reliable', 'moderate', 'unreliable']
        )
        
        # Reset index for merging
        route_stats = route_stats.reset_index()
        
        # Merge route statistics back
        df = df.merge(route_stats, on=['origin', 'destination'], how='left')
        
        # Flight timing risk factors
        df['is_red_eye_flight'] = ((df['departure_hour'] >= 22) | 
                                  (df['departure_hour'] <= 5)).astype(int)
        df['is_dinner_time_flight'] = ((df['departure_hour'] >= 18) & 
                                      (df['departure_hour'] <= 20)).astype(int)
        df['is_rush_hour_flight'] = ((df['departure_hour'] >= 16) & 
                                    (df['departure_hour'] <= 18)).astype(int)
        
        # Flight duration categories
        df['flight_duration_category'] = pd.cut(
            df['flight_duration'],
            bins=[0, 2, 4, 6, float('inf')],
            labels=['short_haul', 'medium_haul', 'long_haul', 'ultra_long_haul']
        )
        
        # Airport hub analysis (simplified - would need real hub data)
        major_hubs = ['JFK', 'LAX', 'ORD', 'DFW', 'ATL', 'DEN', 'SFO', 'SEA']
        df['origin_is_hub'] = df['origin'].isin(major_hubs).astype(int)
        df['destination_is_hub'] = df['destination'].isin(major_hubs).astype(int)
        df['is_hub_to_hub'] = ((df['origin_is_hub'] == 1) & 
                              (df['destination_is_hub'] == 1)).astype(int)
        
        # Aircraft type features
        wide_body_aircraft = ['Boeing 777', 'Boeing 787', 'Airbus A330', 'Airbus A350']
        df['is_wide_body'] = df['aircraft_type'].isin(wide_body_aircraft).astype(int)
        
        # Airline reputation features (simplified)
        premium_airlines = ['American Airlines', 'Delta Air Lines', 'United Airlines']
        budget_airlines = ['Southwest Airlines', 'Spirit Airlines', 'Frontier Airlines']
        
        df['is_premium_airline'] = df['airline'].isin(premium_airlines).astype(int)
        df['is_budget_airline'] = df['airline'].isin(budget_airlines).astype(int)
        
        # Route distance estimation (simplified based on flight duration)
        df['estimated_distance'] = df['flight_duration'] * 500  # Rough approximation
        df['is_transcontinental'] = (df['estimated_distance'] > 2000).astype(int)
        
        logger.info("Flight characteristics features created")
        return df
    
    def create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create economic and pricing features with domain insights.
        
        Domain Insights:
        - Passengers paying below-market rates have higher no-show rates
        - Business class passengers have lower no-show rates
        - Refundable tickets have higher no-show rates (less penalty)
        - Price relative to booking window affects no-show probability
        """
        logger.info("Creating economic features...")
        
        # Price comparison features
        df['route_price_percentile'] = df.groupby(['origin', 'destination'])['ticket_price'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # Price vs. market comparisons
        df['price_vs_route_avg'] = df['ticket_price'] / df['route_avg_price']
        df['price_vs_route_median'] = df['ticket_price'] / df.groupby(['origin', 'destination'])['ticket_price'].transform('median')
        
        # Price categories
        df['is_discount_ticket'] = (df['route_price_percentile'] <= 0.25).astype(int)
        df['is_premium_ticket'] = (df['route_price_percentile'] >= 0.75).astype(int)
        df['is_below_market'] = (df['price_vs_route_avg'] < 0.8).astype(int)
        df['is_above_market'] = (df['price_vs_route_avg'] > 1.2).astype(int)
        
        # Seat class economic impact
        seat_class_values = {
            'economy': 1,
            'premium_economy': 2,
            'business': 3,
            'first': 4
        }
        df['seat_class_value'] = df['seat_class'].map(seat_class_values)
        
        # Price per flight hour
        df['price_per_hour'] = df['ticket_price'] / df['flight_duration']
        df['price_per_hour_percentile'] = df['price_per_hour'].rank(pct=True)
        
        # Advance booking price patterns
        df['price_for_advance_booking'] = df['ticket_price'] / (df['days_to_departure'] + 1)
        
        # Economic segments
        df['economic_segment'] = pd.cut(
            df['ticket_price'],
            bins=[0, 200, 500, 1000, float('inf')],
            labels=['budget', 'economy', 'premium', 'luxury']
        )
        
        # Price sensitivity indicators
        df['likely_price_sensitive'] = (
            (df['is_discount_ticket'] == 1) | 
            (df['economic_segment'] == 'budget')
        ).astype(int)
        
        # Refundability simulation (would need real data)
        # Higher-priced tickets are more likely to be refundable
        df['likely_refundable'] = (
            (df['seat_class_value'] >= 3) | 
            (df['price_vs_route_avg'] > 1.5)
        ).astype(int)
        
        # Corporate travel indicators
        df['likely_corporate_travel'] = (
            (df['seat_class_value'] >= 2) & 
            (df['is_business_travel_pattern'] == 1)
        ).astype(int)
        
        logger.info("Economic features created")
        return df
    
    def create_domain_specific_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific risk features based on airline industry insights.
        
        Domain Insights:
        - Combination of factors create compound risk
        - Certain passenger-flight combinations are extremely high risk
        - Weather and external factors affect no-show rates
        """
        logger.info("Creating domain-specific risk features...")
        
        # High-risk combination features
        df['panic_booking_early_flight'] = (
            (df['is_panic_booking'] == 1) & 
            (df['is_red_eye_risk'] == 1)
        ).astype(int)
        
        df['last_minute_expensive'] = (
            (df['is_last_minute'] == 1) & 
            (df['is_premium_ticket'] == 1)
        ).astype(int)
        
        df['frequent_flyer_unusual_pattern'] = (
            (df['is_frequent_flyer'] == 1) & 
            (df['days_to_departure'] < df['avg_advance_booking'] * 0.5)
        ).astype(int)
        
        df['budget_last_minute'] = (
            (df['is_discount_ticket'] == 1) & 
            (df['is_last_minute'] == 1)
        ).astype(int)
        
        # Weather risk factors
        df['winter_early_morning'] = (
            (df['is_winter_weather'] == 1) & 
            (df['is_early_morning_risk'] == 1)
        ).astype(int)
        
        df['holiday_travel_risk'] = (
            (df['is_major_holiday'] == 1) & 
            (df['is_last_minute'] == 1)
        ).astype(int)
        
        # Passenger reliability score
        df['passenger_reliability_score'] = (
            df['is_frequent_flyer'] * 2 +
            df['is_consistent_booker'] * 1 +
            (1 - df['historical_no_show_rate']) * 3 +
            df['is_premium_airline'] * 1
        )
        
        # Flight convenience score
        df['flight_convenience_score'] = (
            (5 - df['sleep_in_risk_score']) +
            (df['route_volume'] / df['route_volume'].max() * 3) +
            df['is_hub_to_hub'] * 2 +
            (1 - df['is_red_eye_flight']) * 2
        )
        
        # Overall risk score
        df['composite_risk_score'] = (
            df['sleep_in_risk_score'] * 0.3 +
            df['historical_no_show_rate'] * 0.25 +
            (df['is_last_minute'] * 2) * 0.2 +
            (df['route_no_show_rate'] * 3) * 0.15 +
            (df['is_discount_ticket'] * 1) * 0.1
        )
        
        # Risk categories
        df['risk_category'] = pd.cut(
            df['composite_risk_score'],
            bins=[0, 0.5, 1.0, 1.5, float('inf')],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        logger.info("Domain-specific risk features created")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different categories."""
        logger.info("Creating interaction features...")
        
        # Time-based interactions
        df['weekend_early_morning'] = (
            df['is_weekend_departure'] & 
            df['is_early_morning_risk']
        ).astype(int)
        
        df['monday_red_eye'] = (
            (df['departure_day_of_week'] == 0) & 
            df['is_red_eye_flight']
        ).astype(int)
        
        # Passenger-timing interactions
        df['new_passenger_last_minute'] = (
            df['is_new_passenger'] & 
            df['is_last_minute']
        ).astype(int)
        
        df['frequent_flyer_discount'] = (
            df['is_frequent_flyer'] & 
            df['is_discount_ticket']
        ).astype(int)
        
        # Route-timing interactions
        df['popular_route_last_minute'] = (
            (df['route_popularity'] == 'very_popular') & 
            df['is_last_minute']
        ).astype(int)
        
        df['hub_early_morning'] = (
            df['is_hub_to_hub'] & 
            df['is_early_morning_risk']
        ).astype(int)
        
        logger.info("Interaction features created")
        return df
    
    def create_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all advanced features in the correct order."""
        logger.info("Starting advanced feature engineering pipeline...")
        
        # Ensure we have the required columns
        required_columns = ['passenger_id', 'booking_date', 'departure_date', 'no_show']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create features in order (some depend on others)
        df = self.create_advanced_temporal_features(df)
        df = self.create_passenger_behavior_features(df)
        df = self.create_flight_characteristics_features(df)
        df = self.create_economic_features(df)
        df = self.create_domain_specific_risk_features(df)
        df = self.create_interaction_features(df)
        
        logger.info("Advanced feature engineering completed")
        logger.info(f"Dataset shape after feature engineering: {df.shape}")
        
        return df
    
    def get_feature_importance_insights(self) -> Dict[str, str]:
        """Return domain insights about feature importance."""
        return {
            'high_impact_features': [
                'is_same_day_booking',
                'sleep_in_risk_score', 
                'historical_no_show_rate',
                'composite_risk_score',
                'is_red_eye_flight'
            ],
            'insights': {
                'temporal': 'Same-day bookings have 3x higher no-show rates. Early morning flights (before 6 AM) have highest risk.',
                'passenger': 'Historical no-show rate is the strongest predictor. Frequent flyers are more reliable.',
                'flight': 'Red-eye flights and unpopular routes have higher no-show rates.',
                'economic': 'Discount tickets and below-market prices correlate with higher no-show rates.',
                'weather': 'Winter early morning flights are highest risk combinations.',
                'interactions': 'Last-minute bookings combined with early flights create extreme risk.'
            }
        }


def main():
    """Demonstrate advanced feature engineering on sample data."""
    
    # Use pathlib for cross-platform compatibility
    PROJECT_ROOT = Path(__file__).parent.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'airline_bookings.csv'
    
    # Load the generated dataset
    df = pd.read_csv(INPUT_PATH)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original no-show rate: {df['no_show'].mean():.3%}")
    
    # Initialize advanced feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Create all advanced features
    df_enhanced = feature_engineer.create_all_advanced_features(df)
    
    print(f"\nEnhanced dataset shape: {df_enhanced.shape}")
    print(f"New features created: {df_enhanced.shape[1] - df.shape[1]}")
    
    # Show feature importance insights
    insights = feature_engineer.get_feature_importance_insights()
    print("\n=== FEATURE IMPORTANCE INSIGHTS ===")
    for category, insight in insights['insights'].items():
        print(f"{category.upper()}: {insight}")
    
    # Save enhanced dataset
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'airline_data_enhanced.csv'
    
    # Create directories if they don't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df_enhanced.to_csv(OUTPUT_PATH, index=False)
    print(f"\nEnhanced dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()