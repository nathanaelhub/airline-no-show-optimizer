import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    def __init__(self, n_records: int = 10000, no_show_rate: float = 0.08, random_seed: int = 42):
        """
        Initialize synthetic airline booking data generator.
        
        Args:
            n_records: Number of booking records to generate
            no_show_rate: Target no-show rate (between 0.05-0.10)
            random_seed: Random seed for reproducibility
        """
        self.n_records = n_records
        self.no_show_rate = no_show_rate
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define realistic data patterns
        self.airlines = ['American Airlines', 'Delta Air Lines', 'United Airlines', 'Southwest Airlines', 
                        'JetBlue Airways', 'Alaska Airlines', 'Spirit Airlines', 'Frontier Airlines']
        
        self.airports = {
            'JFK': 'New York', 'LAX': 'Los Angeles', 'ORD': 'Chicago', 'DFW': 'Dallas',
            'DEN': 'Denver', 'SFO': 'San Francisco', 'SEA': 'Seattle', 'LAS': 'Las Vegas',
            'MIA': 'Miami', 'MCO': 'Orlando', 'PHX': 'Phoenix', 'BOS': 'Boston',
            'MSP': 'Minneapolis', 'DTW': 'Detroit', 'PHL': 'Philadelphia', 'LGA': 'New York',
            'FLL': 'Fort Lauderdale', 'BWI': 'Baltimore', 'SLC': 'Salt Lake City', 'SAN': 'San Diego'
        }
        
        self.aircraft_types = {
            'Boeing 737': {'capacity': 180, 'route_type': 'domestic'},
            'Boeing 757': {'capacity': 220, 'route_type': 'domestic'},
            'Boeing 767': {'capacity': 280, 'route_type': 'international'},
            'Boeing 777': {'capacity': 350, 'route_type': 'international'},
            'Boeing 787': {'capacity': 330, 'route_type': 'international'},
            'Airbus A320': {'capacity': 180, 'route_type': 'domestic'},
            'Airbus A321': {'capacity': 220, 'route_type': 'domestic'},
            'Airbus A330': {'capacity': 300, 'route_type': 'international'},
            'Airbus A350': {'capacity': 350, 'route_type': 'international'}
        }
        
        self.seat_classes = ['economy', 'premium_economy', 'business', 'first']
        
        # Define passenger archetypes with different no-show probabilities
        self.passenger_types = {
            'business_frequent': {'no_show_prob': 0.12, 'advance_booking': (1, 14), 'price_sensitivity': 'low'},
            'business_occasional': {'no_show_prob': 0.08, 'advance_booking': (7, 30), 'price_sensitivity': 'medium'},
            'leisure_planner': {'no_show_prob': 0.05, 'advance_booking': (30, 120), 'price_sensitivity': 'high'},
            'leisure_spontaneous': {'no_show_prob': 0.15, 'advance_booking': (1, 7), 'price_sensitivity': 'medium'},
            'family_traveler': {'no_show_prob': 0.06, 'advance_booking': (21, 90), 'price_sensitivity': 'high'},
            'student': {'no_show_prob': 0.10, 'advance_booking': (14, 60), 'price_sensitivity': 'high'},
            'senior': {'no_show_prob': 0.04, 'advance_booking': (30, 120), 'price_sensitivity': 'medium'}
        }
        
    def generate_passenger_demographics(self) -> pd.DataFrame:
        """Generate passenger demographic data."""
        logger.info("Generating passenger demographics...")
        
        # Generate passenger IDs
        passenger_ids = [f"P{str(i+1).zfill(8)}" for i in range(self.n_records)]
        
        # Generate passenger types
        passenger_types = np.random.choice(
            list(self.passenger_types.keys()),
            size=self.n_records,
            p=[0.15, 0.10, 0.25, 0.10, 0.20, 0.08, 0.12]  # Realistic distribution
        )
        
        # Generate ages based on passenger type
        ages = []
        for ptype in passenger_types:
            if ptype == 'business_frequent':
                age = np.random.normal(45, 10)
            elif ptype == 'business_occasional':
                age = np.random.normal(40, 12)
            elif ptype == 'leisure_planner':
                age = np.random.normal(35, 15)
            elif ptype == 'leisure_spontaneous':
                age = np.random.normal(28, 8)
            elif ptype == 'family_traveler':
                age = np.random.normal(38, 10)
            elif ptype == 'student':
                age = np.random.normal(22, 3)
            else:  # senior
                age = np.random.normal(68, 8)
            
            ages.append(max(18, min(85, int(age))))
        
        # Generate genders
        genders = np.random.choice(['M', 'F'], size=self.n_records, p=[0.52, 0.48])
        
        # Generate membership status
        membership_levels = []
        for ptype in passenger_types:
            if ptype in ['business_frequent', 'business_occasional']:
                level = np.random.choice(['None', 'Silver', 'Gold', 'Platinum'], p=[0.2, 0.3, 0.3, 0.2])
            elif ptype == 'senior':
                level = np.random.choice(['None', 'Silver', 'Gold', 'Platinum'], p=[0.4, 0.3, 0.2, 0.1])
            else:
                level = np.random.choice(['None', 'Silver', 'Gold', 'Platinum'], p=[0.7, 0.2, 0.08, 0.02])
            membership_levels.append(level)
        
        demographics_df = pd.DataFrame({
            'passenger_id': passenger_ids,
            'passenger_type': passenger_types,
            'age': ages,
            'gender': genders,
            'membership_level': membership_levels
        })
        
        logger.info(f"Generated demographics for {len(demographics_df)} passengers")
        return demographics_df
    
    def generate_booking_characteristics(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate booking characteristics with realistic patterns."""
        logger.info("Generating booking characteristics...")
        
        bookings = []
        
        for idx, row in demographics_df.iterrows():
            passenger_type = row['passenger_type']
            passenger_config = self.passenger_types[passenger_type]
            
            # Generate booking date (last 12 months)
            booking_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            
            # Generate advance booking period based on passenger type
            advance_days = np.random.randint(*passenger_config['advance_booking'])
            departure_date = booking_date + timedelta(days=advance_days)
            
            # Generate route
            origin = np.random.choice(list(self.airports.keys()))
            destination = np.random.choice([k for k in self.airports.keys() if k != origin])
            
            # Generate airline (some passenger types prefer certain airlines)
            if passenger_type in ['business_frequent', 'business_occasional']:
                airline = np.random.choice(['American Airlines', 'Delta Air Lines', 'United Airlines'], 
                                         p=[0.4, 0.35, 0.25])
            elif passenger_type == 'student':
                airline = np.random.choice(['Southwest Airlines', 'Spirit Airlines', 'Frontier Airlines'], 
                                         p=[0.5, 0.3, 0.2])
            else:
                airline = np.random.choice(self.airlines)
            
            # Generate aircraft type
            aircraft = np.random.choice(list(self.aircraft_types.keys()))
            
            # Generate seat class based on passenger type
            if passenger_type in ['business_frequent', 'business_occasional']:
                seat_class = np.random.choice(['economy', 'premium_economy', 'business', 'first'],
                                            p=[0.3, 0.2, 0.4, 0.1])
            elif passenger_type == 'student':
                seat_class = 'economy'
            else:
                seat_class = np.random.choice(['economy', 'premium_economy', 'business'],
                                            p=[0.85, 0.12, 0.03])
            
            # Generate flight duration based on route
            flight_duration = np.random.uniform(1.5, 8.0)  # hours
            
            # Generate ticket price based on multiple factors
            base_price = self._calculate_ticket_price(
                advance_days, seat_class, flight_duration, 
                departure_date, passenger_config['price_sensitivity']
            )
            
            # Generate flight number
            flight_number = f"{airline[:2].upper()}{np.random.randint(1000, 9999)}"
            
            bookings.append({
                'passenger_id': row['passenger_id'],
                'booking_date': booking_date,
                'departure_date': departure_date,
                'arrival_date': departure_date + timedelta(hours=flight_duration),
                'flight_number': flight_number,
                'airline': airline,
                'origin': origin,
                'destination': destination,
                'aircraft_type': aircraft,
                'seat_class': seat_class,
                'ticket_price': base_price,
                'flight_duration': flight_duration,
                'advance_booking_days': advance_days
            })
        
        bookings_df = pd.DataFrame(bookings)
        logger.info(f"Generated {len(bookings_df)} booking records")
        return bookings_df
    
    def _calculate_ticket_price(self, advance_days: int, seat_class: str, duration: float, 
                              departure_date: datetime, price_sensitivity: str) -> float:
        """Calculate realistic ticket price based on multiple factors."""
        
        # Base price by seat class
        base_prices = {
            'economy': 200,
            'premium_economy': 400,
            'business': 800,
            'first': 1500
        }
        
        base_price = base_prices[seat_class]
        
        # Duration factor
        duration_factor = 1 + (duration - 2) * 0.1
        
        # Advance booking factor
        if advance_days <= 3:
            advance_factor = 1.5  # Last minute premium
        elif advance_days <= 7:
            advance_factor = 1.3
        elif advance_days <= 21:
            advance_factor = 1.0
        else:
            advance_factor = 0.8  # Early booking discount
        
        # Seasonal factor
        month = departure_date.month
        if month in [12, 1, 7, 8]:  # Peak season
            seasonal_factor = 1.3
        elif month in [3, 4, 5, 9, 10]:  # Shoulder season
            seasonal_factor = 1.1
        else:  # Off season
            seasonal_factor = 0.9
        
        # Day of week factor
        if departure_date.weekday() in [4, 6]:  # Friday, Sunday
            dow_factor = 1.2
        else:
            dow_factor = 1.0
        
        # Price sensitivity adjustment
        if price_sensitivity == 'high':
            price_adjustment = np.random.uniform(0.7, 1.1)
        elif price_sensitivity == 'medium':
            price_adjustment = np.random.uniform(0.9, 1.3)
        else:  # low
            price_adjustment = np.random.uniform(1.0, 1.5)
        
        final_price = base_price * duration_factor * advance_factor * seasonal_factor * dow_factor * price_adjustment
        
        # Add some random variation
        final_price *= np.random.uniform(0.9, 1.1)
        
        return round(final_price, 2)
    
    def generate_no_show_patterns(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Generate no-show patterns based on passenger and booking characteristics."""
        logger.info("Generating no-show patterns...")
        
        no_shows = []
        
        for idx, row in combined_df.iterrows():
            passenger_type = row['passenger_type']
            base_no_show_prob = self.passenger_types[passenger_type]['no_show_prob']
            
            # Adjust probability based on various factors
            prob_adjustments = 1.0
            
            # Advance booking factor
            if row['advance_booking_days'] <= 1:
                prob_adjustments *= 1.8  # Same day bookings more likely to no-show
            elif row['advance_booking_days'] <= 3:
                prob_adjustments *= 1.4
            elif row['advance_booking_days'] >= 60:
                prob_adjustments *= 0.8  # Far advance bookings less likely to no-show
            
            # Ticket price factor (higher prices = lower no-show)
            price_percentile = np.percentile(combined_df['ticket_price'], 75)
            if row['ticket_price'] > price_percentile:
                prob_adjustments *= 0.7
            elif row['ticket_price'] < np.percentile(combined_df['ticket_price'], 25):
                prob_adjustments *= 1.3
            
            # Seat class factor
            if row['seat_class'] in ['business', 'first']:
                prob_adjustments *= 0.6
            elif row['seat_class'] == 'premium_economy':
                prob_adjustments *= 0.8
            
            # Membership level factor
            membership_factors = {
                'None': 1.0,
                'Silver': 0.8,
                'Gold': 0.6,
                'Platinum': 0.4
            }
            prob_adjustments *= membership_factors[row['membership_level']]
            
            # Age factor
            if row['age'] < 25:
                prob_adjustments *= 1.2  # Young passengers more likely to no-show
            elif row['age'] > 65:
                prob_adjustments *= 0.7  # Older passengers less likely to no-show
            
            # Day of week factor
            if row['departure_date'].weekday() == 0:  # Monday
                prob_adjustments *= 1.1
            elif row['departure_date'].weekday() == 6:  # Sunday
                prob_adjustments *= 0.9
            
            # Time of day factor
            departure_hour = row['departure_date'].hour
            if departure_hour < 6:  # Very early flights
                prob_adjustments *= 1.4
            elif departure_hour >= 22:  # Late night flights
                prob_adjustments *= 1.2
            
            # Calculate final probability
            final_prob = base_no_show_prob * prob_adjustments
            final_prob = max(0.01, min(0.3, final_prob))  # Clamp between 1% and 30%
            
            # Generate no-show decision
            no_show = np.random.random() < final_prob
            no_shows.append(no_show)
        
        combined_df['no_show'] = no_shows
        
        actual_no_show_rate = combined_df['no_show'].mean()
        logger.info(f"Generated no-show patterns. Actual rate: {actual_no_show_rate:.3f}")
        
        return combined_df
    
    def add_seasonal_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal booking patterns and variations."""
        logger.info("Adding seasonal variations...")
        
        # Add seasonal indicators
        df['booking_month'] = df['booking_date'].dt.month
        df['departure_month'] = df['departure_date'].dt.month
        df['booking_quarter'] = df['booking_date'].dt.quarter
        df['departure_quarter'] = df['departure_date'].dt.quarter
        
        # Add holiday season indicators
        df['is_holiday_season'] = df['departure_month'].isin([12, 1, 7, 8]).astype(int)
        df['is_summer_travel'] = df['departure_month'].isin([6, 7, 8]).astype(int)
        df['is_winter_travel'] = df['departure_month'].isin([12, 1, 2]).astype(int)
        
        # Add day of week indicators
        df['booking_day_of_week'] = df['booking_date'].dt.dayofweek
        df['departure_day_of_week'] = df['departure_date'].dt.dayofweek
        df['is_weekend_departure'] = df['departure_day_of_week'].isin([5, 6]).astype(int)
        df['is_weekday_departure'] = (~df['departure_day_of_week'].isin([5, 6])).astype(int)
        
        # Add time-based features
        df['departure_hour'] = df['departure_date'].dt.hour
        df['booking_hour'] = df['booking_date'].dt.hour
        
        # Categorize departure times
        df['departure_time_category'] = pd.cut(
            df['departure_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['early_morning', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        logger.info("Added seasonal variations and time-based features")
        return df
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete synthetic airline booking dataset."""
        logger.info(f"Starting generation of {self.n_records} synthetic booking records...")
        
        # Step 1: Generate passenger demographics
        demographics_df = self.generate_passenger_demographics()
        
        # Step 2: Generate booking characteristics
        bookings_df = self.generate_booking_characteristics(demographics_df)
        
        # Step 3: Combine demographics and bookings
        combined_df = pd.merge(demographics_df, bookings_df, on='passenger_id')
        
        # Step 4: Generate no-show patterns
        combined_df = self.generate_no_show_patterns(combined_df)
        
        # Step 5: Add seasonal variations
        combined_df = self.add_seasonal_variations(combined_df)
        
        # Step 6: Final data cleaning and formatting
        combined_df = self._finalize_dataset(combined_df)
        
        logger.info(f"Dataset generation complete. Shape: {combined_df.shape}")
        logger.info(f"Final no-show rate: {combined_df['no_show'].mean():.3f}")
        
        return combined_df
    
    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning and formatting of the dataset."""
        
        # Convert boolean no_show to integer
        df['no_show'] = df['no_show'].astype(int)
        
        # Round numerical columns
        df['ticket_price'] = df['ticket_price'].round(2)
        df['flight_duration'] = df['flight_duration'].round(1)
        
        # Ensure proper data types
        df['booking_date'] = pd.to_datetime(df['booking_date'])
        df['departure_date'] = pd.to_datetime(df['departure_date'])
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        
        # Reorder columns for better readability
        column_order = [
            'passenger_id', 'booking_date', 'departure_date', 'arrival_date',
            'flight_number', 'airline', 'origin', 'destination', 'aircraft_type',
            'seat_class', 'ticket_price', 'flight_duration', 'advance_booking_days',
            'passenger_type', 'age', 'gender', 'membership_level',
            'booking_month', 'departure_month', 'booking_quarter', 'departure_quarter',
            'is_holiday_season', 'is_summer_travel', 'is_winter_travel',
            'booking_day_of_week', 'departure_day_of_week', 'is_weekend_departure',
            'departure_hour', 'booking_hour', 'departure_time_category',
            'no_show'
        ]
        
        df = df[column_order]
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """Save the generated dataset to CSV."""
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
        
        # Print summary statistics
        print("\n=== DATASET SUMMARY ===")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['booking_date'].min().date()} to {df['booking_date'].max().date()}")
        print(f"No-show rate: {df['no_show'].mean():.3%}")
        print(f"Average ticket price: ${df['ticket_price'].mean():.2f}")
        print(f"Unique passengers: {df['passenger_id'].nunique():,}")
        print(f"Unique routes: {df[['origin', 'destination']].drop_duplicates().shape[0]:,}")
        print(f"Airlines: {', '.join(df['airline'].unique())}")
        
        print("\n=== NO-SHOW RATES BY SEGMENT ===")
        print(df.groupby('passenger_type')['no_show'].agg(['count', 'mean']).round(3))
        
        print("\n=== BOOKING PATTERNS ===")
        print(f"Average advance booking: {df['advance_booking_days'].mean():.1f} days")
        print(f"Seat class distribution:")
        print(df['seat_class'].value_counts(normalize=True).round(3))


def main():
    """Generate and save synthetic airline booking data."""
    
    # Configuration
    N_RECORDS = 12000
    NO_SHOW_RATE = 0.078  # Target 7.8% no-show rate
    
    # Use pathlib for cross-platform compatibility
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'airline_bookings.csv'
    
    # Create directories if they don't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    generator = SyntheticDataGenerator(
        n_records=N_RECORDS,
        no_show_rate=NO_SHOW_RATE,
        random_seed=42
    )
    
    # Create dataset
    dataset = generator.generate_complete_dataset()
    
    # Save dataset
    generator.save_dataset(dataset, str(OUTPUT_PATH))
    
    print(f"\nSynthetic airline booking dataset generated successfully!")
    print(f"File saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()