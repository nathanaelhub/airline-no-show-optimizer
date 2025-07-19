import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RevenueOptimizer:
    def __init__(self, denied_boarding_cost: float = 800, 
                 empty_seat_cost: float = 200):
        self.denied_boarding_cost = denied_boarding_cost
        self.empty_seat_cost = empty_seat_cost
        self.optimization_results = {}
        
    def calculate_expected_revenue(self, bookings: int, capacity: int, 
                                 no_show_probabilities: np.ndarray, 
                                 ticket_prices: np.ndarray) -> float:
        """Calculate expected revenue given bookings and no-show probabilities."""
        
        # Simulate outcomes
        n_simulations = 10000
        total_revenue = 0
        
        for _ in range(n_simulations):
            # Simulate no-shows
            no_shows = np.random.binomial(1, no_show_probabilities)
            actual_passengers = bookings - np.sum(no_shows)
            
            # Calculate revenue
            revenue = np.sum(ticket_prices)  # Total ticket sales
            
            # Subtract costs
            if actual_passengers > capacity:
                # Denied boarding cost
                denied_passengers = actual_passengers - capacity
                revenue -= denied_passengers * self.denied_boarding_cost
            elif actual_passengers < capacity:
                # Empty seat opportunity cost
                empty_seats = capacity - actual_passengers
                revenue -= empty_seats * self.empty_seat_cost
            
            total_revenue += revenue
        
        return total_revenue / n_simulations
    
    def optimize_overbooking_rate(self, flight_data: pd.DataFrame, 
                                 capacity: int, confidence_level: float = 0.95) -> Dict[str, float]:
        """Optimize overbooking rate for a specific flight."""
        logger.info("Optimizing overbooking rate...")
        
        # Extract no-show probabilities and ticket prices
        no_show_probs = flight_data['no_show_probability'].values
        ticket_prices = flight_data['ticket_price'].values
        base_bookings = len(flight_data)
        
        def objective_function(overbooking_rate):
            """Objective function to maximize expected revenue."""
            additional_bookings = int(base_bookings * overbooking_rate[0])
            total_bookings = base_bookings + additional_bookings
            
            # Extend arrays for additional bookings (use mean values)
            extended_no_show_probs = np.concatenate([
                no_show_probs, 
                np.full(additional_bookings, np.mean(no_show_probs))
            ])
            extended_ticket_prices = np.concatenate([
                ticket_prices,
                np.full(additional_bookings, np.mean(ticket_prices))
            ])
            
            expected_revenue = self.calculate_expected_revenue(
                total_bookings, capacity, extended_no_show_probs, extended_ticket_prices
            )
            
            return -expected_revenue  # Negative because we minimize
        
        # Optimize
        result = minimize(
            objective_function, 
            x0=[0.1],  # Initial guess: 10% overbooking
            bounds=[(0, 0.5)],  # Max 50% overbooking
            method='L-BFGS-B'
        )
        
        optimal_rate = result.x[0]
        max_revenue = -result.fun
        
        # Calculate key metrics
        additional_bookings = int(base_bookings * optimal_rate)
        total_bookings = base_bookings + additional_bookings
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            flight_data, capacity, optimal_rate, confidence_level
        )
        
        optimization_result = {
            'optimal_overbooking_rate': optimal_rate,
            'additional_bookings': additional_bookings,
            'total_bookings': total_bookings,
            'expected_revenue': max_revenue,
            'denied_boarding_probability': risk_metrics['denied_boarding_prob'],
            'expected_denied_passengers': risk_metrics['expected_denied'],
            'revenue_improvement': max_revenue - self.calculate_expected_revenue(
                base_bookings, capacity, no_show_probs, ticket_prices
            )
        }
        
        logger.info(f"Optimal overbooking rate: {optimal_rate:.2%}")
        logger.info(f"Expected revenue improvement: ${optimization_result['revenue_improvement']:.2f}")
        
        return optimization_result
    
    def _calculate_risk_metrics(self, flight_data: pd.DataFrame, capacity: int, 
                              overbooking_rate: float, confidence_level: float) -> Dict[str, float]:
        """Calculate risk metrics for the optimization."""
        no_show_probs = flight_data['no_show_probability'].values
        base_bookings = len(flight_data)
        additional_bookings = int(base_bookings * overbooking_rate)
        total_bookings = base_bookings + additional_bookings
        
        # Monte Carlo simulation for risk assessment
        n_simulations = 10000
        denied_boarding_counts = []
        
        for _ in range(n_simulations):
            # Simulate no-shows
            extended_no_show_probs = np.concatenate([
                no_show_probs, 
                np.full(additional_bookings, np.mean(no_show_probs))
            ])
            
            no_shows = np.random.binomial(1, extended_no_show_probs)
            actual_passengers = total_bookings - np.sum(no_shows)
            
            denied_passengers = max(0, actual_passengers - capacity)
            denied_boarding_counts.append(denied_passengers)
        
        denied_boarding_prob = np.mean(np.array(denied_boarding_counts) > 0)
        expected_denied = np.mean(denied_boarding_counts)
        
        return {
            'denied_boarding_prob': denied_boarding_prob,
            'expected_denied': expected_denied
        }
    
    def optimize_fleet_wide(self, flights_data: Dict[str, pd.DataFrame], 
                           capacities: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """Optimize overbooking rates across multiple flights."""
        logger.info("Optimizing fleet-wide overbooking rates...")
        
        fleet_results = {}
        total_revenue_improvement = 0
        
        for flight_id, flight_data in flights_data.items():
            capacity = capacities[flight_id]
            
            try:
                result = self.optimize_overbooking_rate(flight_data, capacity)
                fleet_results[flight_id] = result
                total_revenue_improvement += result['revenue_improvement']
                
            except Exception as e:
                logger.error(f"Error optimizing flight {flight_id}: {e}")
                continue
        
        # Summary statistics
        fleet_summary = {
            'total_flights': len(fleet_results),
            'total_revenue_improvement': total_revenue_improvement,
            'average_overbooking_rate': np.mean([
                result['optimal_overbooking_rate'] 
                for result in fleet_results.values()
            ]),
            'total_additional_bookings': sum([
                result['additional_bookings'] 
                for result in fleet_results.values()
            ])
        }
        
        self.optimization_results = {
            'fleet_results': fleet_results,
            'fleet_summary': fleet_summary
        }
        
        logger.info(f"Fleet-wide optimization completed")
        logger.info(f"Total revenue improvement: ${total_revenue_improvement:.2f}")
        
        return self.optimization_results
    
    def dynamic_pricing_optimization(self, flight_data: pd.DataFrame, 
                                   capacity: int, days_to_departure: int) -> Dict[str, float]:
        """Optimize pricing based on demand and no-show predictions."""
        logger.info("Optimizing dynamic pricing...")
        
        # Get current booking load
        current_bookings = len(flight_data)
        load_factor = current_bookings / capacity
        
        # Estimate demand elasticity based on historical data
        base_price = flight_data['ticket_price'].mean()
        no_show_rate = flight_data['no_show_probability'].mean()
        
        # Price adjustment factors
        time_factor = self._calculate_time_factor(days_to_departure)
        demand_factor = self._calculate_demand_factor(load_factor)
        risk_factor = self._calculate_risk_factor(no_show_rate)
        
        # Calculate optimal price
        optimal_price = base_price * time_factor * demand_factor * risk_factor
        
        # Calculate expected impact
        price_change = (optimal_price - base_price) / base_price
        expected_demand_change = -0.5 * price_change  # Simplified elasticity
        
        pricing_result = {
            'current_price': base_price,
            'optimal_price': optimal_price,
            'price_change_percent': price_change * 100,
            'expected_demand_change_percent': expected_demand_change * 100,
            'time_factor': time_factor,
            'demand_factor': demand_factor,
            'risk_factor': risk_factor,
            'current_load_factor': load_factor,
            'days_to_departure': days_to_departure
        }
        
        logger.info(f"Optimal price: ${optimal_price:.2f} ({price_change:.1%} change)")
        
        return pricing_result
    
    def _calculate_time_factor(self, days_to_departure: int) -> float:
        """Calculate pricing factor based on time to departure."""
        if days_to_departure <= 1:
            return 1.2  # Last-minute premium
        elif days_to_departure <= 7:
            return 1.1  # Week before premium
        elif days_to_departure <= 21:
            return 1.0  # Standard pricing
        else:
            return 0.9  # Early booking discount
    
    def _calculate_demand_factor(self, load_factor: float) -> float:
        """Calculate pricing factor based on current demand."""
        if load_factor >= 0.9:
            return 1.3  # High demand premium
        elif load_factor >= 0.7:
            return 1.1  # Moderate demand
        elif load_factor >= 0.5:
            return 1.0  # Standard pricing
        else:
            return 0.8  # Low demand discount
    
    def _calculate_risk_factor(self, no_show_rate: float) -> float:
        """Calculate pricing factor based on no-show risk."""
        if no_show_rate >= 0.2:
            return 1.1  # High risk premium
        elif no_show_rate >= 0.1:
            return 1.0  # Standard pricing
        else:
            return 0.95  # Low risk discount
    
    def generate_recommendations(self, flight_data: pd.DataFrame, 
                               capacity: int, days_to_departure: int) -> Dict[str, any]:
        """Generate comprehensive optimization recommendations."""
        logger.info("Generating optimization recommendations...")
        
        # Get optimization results
        overbooking_result = self.optimize_overbooking_rate(flight_data, capacity)
        pricing_result = self.dynamic_pricing_optimization(
            flight_data, capacity, days_to_departure
        )
        
        # Risk assessment
        current_load_factor = len(flight_data) / capacity
        risk_level = self._assess_risk_level(
            current_load_factor, 
            overbooking_result['denied_boarding_probability'],
            days_to_departure
        )
        
        recommendations = {
            'overbooking': overbooking_result,
            'pricing': pricing_result,
            'risk_assessment': {
                'risk_level': risk_level,
                'current_load_factor': current_load_factor,
                'recommendations': self._generate_risk_recommendations(risk_level)
            },
            'key_actions': [
                f"Implement {overbooking_result['optimal_overbooking_rate']:.1%} overbooking rate",
                f"Adjust price to ${pricing_result['optimal_price']:.2f}",
                f"Monitor bookings closely due to {risk_level} risk level"
            ]
        }
        
        return recommendations
    
    def _assess_risk_level(self, load_factor: float, denied_boarding_prob: float, 
                          days_to_departure: int) -> str:
        """Assess overall risk level for the flight."""
        risk_score = 0
        
        # Load factor risk
        if load_factor >= 0.9:
            risk_score += 2
        elif load_factor >= 0.7:
            risk_score += 1
        
        # Denied boarding risk
        if denied_boarding_prob >= 0.1:
            risk_score += 2
        elif denied_boarding_prob >= 0.05:
            risk_score += 1
        
        # Time risk
        if days_to_departure <= 1:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_risk_recommendations(self, risk_level: str) -> List[str]:
        """Generate risk-specific recommendations."""
        if risk_level == "HIGH":
            return [
                "Consider reducing overbooking rate",
                "Implement more aggressive passenger incentives",
                "Monitor bookings hourly",
                "Prepare contingency plans for denied boarding"
            ]
        elif risk_level == "MEDIUM":
            return [
                "Monitor bookings closely",
                "Consider moderate pricing adjustments",
                "Have customer service team on standby"
            ]
        else:
            return [
                "Continue current strategy",
                "Consider opportunities for increased overbooking",
                "Monitor for demand changes"
            ]