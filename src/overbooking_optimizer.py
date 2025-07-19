import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from scipy import optimize, stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Cost parameters for overbooking optimization."""
    denied_boarding_cost: float = 1200.0  # Cost per denied boarding passenger
    empty_seat_cost: float = 400.0        # Opportunity cost per empty seat
    volunteer_compensation: float = 800.0  # Cost for voluntary denied boarding
    involuntary_multiplier: float = 1.5   # Multiplier for involuntary denied boarding
    customer_goodwill_cost: float = 200.0 # Long-term customer relationship cost
    rebooking_cost: float = 150.0         # Administrative cost of rebooking


@dataclass
class FlightParameters:
    """Flight-specific parameters."""
    capacity: int = 180
    base_fare: float = 400.0
    load_factor_target: float = 0.95
    route_type: str = "domestic"  # domestic, international
    aircraft_type: str = "narrow_body"


class OverbookingStrategy:
    """Different overbooking strategy implementations."""
    
    @staticmethod
    def conservative(predicted_no_shows: np.ndarray, capacity: int, 
                    confidence_level: float = 0.95) -> int:
        """
        Conservative strategy: Overbook only when very confident.
        Uses lower confidence interval of predicted no-shows.
        """
        # Calculate confidence interval for no-shows
        mean_no_shows = np.mean(predicted_no_shows)
        std_no_shows = np.std(predicted_no_shows)
        
        # Use lower bound of confidence interval
        z_score = stats.norm.ppf(confidence_level)
        lower_bound = max(0, mean_no_shows - z_score * std_no_shows)
        
        # Conservative overbooking: only up to lower bound
        max_overbooking = int(np.floor(lower_bound))
        return min(max_overbooking, int(capacity * 0.05))  # Cap at 5% of capacity
    
    @staticmethod
    def moderate(predicted_no_shows: np.ndarray, capacity: int,
                confidence_level: float = 0.80) -> int:
        """
        Moderate strategy: Balance between revenue and risk.
        Uses expected value with some safety margin.
        """
        mean_no_shows = np.mean(predicted_no_shows)
        std_no_shows = np.std(predicted_no_shows)
        
        # Use moderate confidence level
        z_score = stats.norm.ppf(confidence_level)
        target_no_shows = mean_no_shows - 0.5 * z_score * std_no_shows
        
        max_overbooking = int(np.round(target_no_shows))
        return min(max_overbooking, int(capacity * 0.10))  # Cap at 10% of capacity
    
    @staticmethod
    def aggressive(predicted_no_shows: np.ndarray, capacity: int,
                  confidence_level: float = 0.60) -> int:
        """
        Aggressive strategy: Maximize revenue with higher risk tolerance.
        Uses upper confidence interval.
        """
        mean_no_shows = np.mean(predicted_no_shows)
        std_no_shows = np.std(predicted_no_shows)
        
        # Use upper bound for aggressive strategy
        z_score = stats.norm.ppf(confidence_level)
        upper_bound = mean_no_shows + 0.3 * z_score * std_no_shows
        
        max_overbooking = int(np.ceil(upper_bound))
        return min(max_overbooking, int(capacity * 0.15))  # Cap at 15% of capacity


class OverbookingOptimizer:
    """
    Advanced overbooking optimization algorithm using no-show predictions
    to maximize airline revenue while managing denied boarding costs.
    """
    
    def __init__(self, cost_params: CostParameters = None, 
                 flight_params: FlightParameters = None):
        self.cost_params = cost_params or CostParameters()
        self.flight_params = flight_params or FlightParameters()
        self.optimization_results = {}
        self.strategy = OverbookingStrategy()
        
    def calculate_expected_revenue(self, current_bookings: int, overbooking_level: int,
                                 no_show_probabilities: np.ndarray,
                                 ticket_prices: np.ndarray = None,
                                 n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate expected revenue for a given overbooking level using Monte Carlo simulation.
        
        Args:
            current_bookings: Number of current bookings
            overbooking_level: Additional seats to sell beyond capacity
            no_show_probabilities: Individual passenger no-show probabilities
            ticket_prices: Individual ticket prices (if None, uses base fare)
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with revenue metrics
        """
        if ticket_prices is None:
            ticket_prices = np.full(len(no_show_probabilities), self.flight_params.base_fare)
        
        total_bookings = current_bookings + overbooking_level
        capacity = self.flight_params.capacity
        
        # Extend arrays for overbooking passengers
        if overbooking_level > 0:
            # New passengers have average no-show probability
            avg_no_show_prob = np.mean(no_show_probabilities)
            extended_no_show_probs = np.concatenate([
                no_show_probabilities,
                np.full(overbooking_level, avg_no_show_prob)
            ])
            extended_prices = np.concatenate([
                ticket_prices,
                np.full(overbooking_level, self.flight_params.base_fare)
            ])
        else:
            extended_no_show_probs = no_show_probabilities
            extended_prices = ticket_prices
        
        revenues = []
        denied_boardings = []
        empty_seats = []
        
        for _ in range(n_simulations):
            # Simulate no-shows
            no_shows = np.random.binomial(1, extended_no_show_probs)
            actual_passengers = total_bookings - np.sum(no_shows)
            
            # Calculate revenue components
            ticket_revenue = np.sum(extended_prices)
            
            # Denied boarding costs
            if actual_passengers > capacity:
                denied_count = actual_passengers - capacity
                denied_cost = self._calculate_denied_boarding_cost(denied_count)
                denied_boardings.append(denied_count)
                empty_seats.append(0)
            else:
                denied_cost = 0
                denied_boardings.append(0)
                empty_seats.append(capacity - actual_passengers)
            
            # Empty seat opportunity cost
            empty_seat_count = max(0, capacity - actual_passengers)
            empty_seat_cost = empty_seat_count * self.cost_params.empty_seat_cost
            
            # Net revenue
            net_revenue = ticket_revenue - denied_cost - empty_seat_cost
            revenues.append(net_revenue)
        
        return {
            'expected_revenue': np.mean(revenues),
            'revenue_std': np.std(revenues),
            'revenue_5th_percentile': np.percentile(revenues, 5),
            'revenue_95th_percentile': np.percentile(revenues, 95),
            'avg_denied_boardings': np.mean(denied_boardings),
            'denied_boarding_probability': np.mean(np.array(denied_boardings) > 0),
            'avg_empty_seats': np.mean(empty_seats),
            'ticket_revenue': np.mean([np.sum(extended_prices)] * n_simulations),
            'load_factor': (capacity - np.mean(empty_seats)) / capacity
        }
    
    def _calculate_denied_boarding_cost(self, denied_count: int) -> float:
        """Calculate total cost of denied boardings."""
        if denied_count <= 0:
            return 0
        
        # Assume some passengers volunteer for compensation
        volunteer_rate = 0.3  # 30% of denied passengers volunteer
        volunteers = min(denied_count, int(denied_count * volunteer_rate))
        involuntary = denied_count - volunteers
        
        # Calculate costs
        volunteer_cost = volunteers * self.cost_params.volunteer_compensation
        involuntary_cost = involuntary * (
            self.cost_params.denied_boarding_cost * 
            self.cost_params.involuntary_multiplier
        )
        
        # Additional costs
        goodwill_cost = denied_count * self.cost_params.customer_goodwill_cost
        rebooking_cost = denied_count * self.cost_params.rebooking_cost
        
        return volunteer_cost + involuntary_cost + goodwill_cost + rebooking_cost
    
    def optimize_overbooking_level(self, current_bookings: int,
                                 no_show_probabilities: np.ndarray,
                                 ticket_prices: np.ndarray = None,
                                 max_overbooking: int = None) -> Dict[str, Any]:
        """
        Find optimal overbooking level to maximize expected revenue.
        
        Args:
            current_bookings: Current number of bookings
            no_show_probabilities: Individual passenger no-show probabilities
            ticket_prices: Individual ticket prices
            max_overbooking: Maximum overbooking level to consider
            
        Returns:
            Optimization results
        """
        logger.info("Optimizing overbooking level...")
        
        if max_overbooking is None:
            max_overbooking = min(50, int(self.flight_params.capacity * 0.2))
        
        # Test different overbooking levels
        overbooking_levels = range(0, max_overbooking + 1)
        results = []
        
        for level in overbooking_levels:
            revenue_metrics = self.calculate_expected_revenue(
                current_bookings, level, no_show_probabilities, ticket_prices
            )
            revenue_metrics['overbooking_level'] = level
            results.append(revenue_metrics)
        
        # Find optimal level
        revenues = [r['expected_revenue'] for r in results]
        optimal_idx = np.argmax(revenues)
        optimal_result = results[optimal_idx]
        
        # Calculate baseline (no overbooking)
        baseline_result = results[0]  # overbooking_level = 0
        
        optimization_result = {
            'optimal_overbooking_level': optimal_result['overbooking_level'],
            'optimal_expected_revenue': optimal_result['expected_revenue'],
            'baseline_revenue': baseline_result['expected_revenue'],
            'revenue_improvement': optimal_result['expected_revenue'] - baseline_result['expected_revenue'],
            'denied_boarding_probability': optimal_result['denied_boarding_probability'],
            'expected_denied_boardings': optimal_result['avg_denied_boardings'],
            'load_factor': optimal_result['load_factor'],
            'all_results': results,
            'cost_parameters': self.cost_params,
            'flight_parameters': self.flight_params
        }
        
        logger.info(f"Optimal overbooking level: {optimal_result['overbooking_level']}")
        logger.info(f"Revenue improvement: ${optimization_result['revenue_improvement']:,.2f}")
        
        return optimization_result
    
    def compare_strategies(self, current_bookings: int,
                          no_show_probabilities: np.ndarray,
                          ticket_prices: np.ndarray = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare different overbooking strategies.
        
        Returns:
            Results for each strategy
        """
        logger.info("Comparing overbooking strategies...")
        
        capacity = self.flight_params.capacity
        strategies = {
            'conservative': self.strategy.conservative(no_show_probabilities, capacity),
            'moderate': self.strategy.moderate(no_show_probabilities, capacity),
            'aggressive': self.strategy.aggressive(no_show_probabilities, capacity),
        }
        
        # Add optimal strategy
        optimal_result = self.optimize_overbooking_level(
            current_bookings, no_show_probabilities, ticket_prices
        )
        strategies['optimal'] = optimal_result['optimal_overbooking_level']
        
        # Evaluate each strategy
        strategy_results = {}
        for strategy_name, overbooking_level in strategies.items():
            revenue_metrics = self.calculate_expected_revenue(
                current_bookings, overbooking_level, no_show_probabilities, ticket_prices
            )
            
            strategy_results[strategy_name] = {
                'overbooking_level': overbooking_level,
                'expected_revenue': revenue_metrics['expected_revenue'],
                'denied_boarding_probability': revenue_metrics['denied_boarding_probability'],
                'expected_denied_boardings': revenue_metrics['avg_denied_boardings'],
                'load_factor': revenue_metrics['load_factor'],
                'revenue_std': revenue_metrics['revenue_std']
            }
        
        # Add baseline (no overbooking)
        baseline_metrics = self.calculate_expected_revenue(
            current_bookings, 0, no_show_probabilities, ticket_prices
        )
        strategy_results['baseline'] = {
            'overbooking_level': 0,
            'expected_revenue': baseline_metrics['expected_revenue'],
            'denied_boarding_probability': 0,
            'expected_denied_boardings': 0,
            'load_factor': baseline_metrics['load_factor'],
            'revenue_std': baseline_metrics['revenue_std']
        }
        
        return strategy_results
    
    def sensitivity_analysis(self, current_bookings: int,
                           no_show_probabilities: np.ndarray,
                           ticket_prices: np.ndarray = None,
                           cost_variations: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on cost parameters.
        
        Args:
            current_bookings: Current bookings
            no_show_probabilities: No-show probabilities
            ticket_prices: Ticket prices
            cost_variations: Dictionary of cost parameter variations to test
            
        Returns:
            Sensitivity analysis results
        """
        logger.info("Performing sensitivity analysis...")
        
        if cost_variations is None:
            cost_variations = {
                'denied_boarding_cost': [800, 1000, 1200, 1500, 2000],
                'empty_seat_cost': [200, 300, 400, 500, 600],
                'volunteer_compensation': [400, 600, 800, 1000, 1200]
            }
        
        sensitivity_results = {}
        base_result = self.optimize_overbooking_level(
            current_bookings, no_show_probabilities, ticket_prices
        )
        
        for param_name, param_values in cost_variations.items():
            param_results = []
            
            for param_value in param_values:
                # Create modified cost parameters
                modified_costs = CostParameters(
                    denied_boarding_cost=self.cost_params.denied_boarding_cost,
                    empty_seat_cost=self.cost_params.empty_seat_cost,
                    volunteer_compensation=self.cost_params.volunteer_compensation,
                    involuntary_multiplier=self.cost_params.involuntary_multiplier,
                    customer_goodwill_cost=self.cost_params.customer_goodwill_cost,
                    rebooking_cost=self.cost_params.rebooking_cost
                )
                setattr(modified_costs, param_name, param_value)
                
                # Temporarily update cost parameters
                original_costs = self.cost_params
                self.cost_params = modified_costs
                
                # Optimize with modified costs
                result = self.optimize_overbooking_level(
                    current_bookings, no_show_probabilities, ticket_prices
                )
                
                param_results.append({
                    'parameter_value': param_value,
                    'optimal_overbooking_level': result['optimal_overbooking_level'],
                    'expected_revenue': result['optimal_expected_revenue'],
                    'revenue_improvement': result['revenue_improvement'],
                    'denied_boarding_probability': result['denied_boarding_probability']
                })
                
                # Restore original costs
                self.cost_params = original_costs
            
            sensitivity_results[param_name] = param_results
        
        return {
            'sensitivity_results': sensitivity_results,
            'base_result': base_result,
            'cost_variations': cost_variations
        }
    
    def generate_optimization_report(self, current_bookings: int,
                                   no_show_probabilities: np.ndarray,
                                   ticket_prices: np.ndarray = None) -> str:
        """Generate comprehensive optimization report."""
        
        # Run all analyses
        strategy_comparison = self.compare_strategies(
            current_bookings, no_show_probabilities, ticket_prices
        )
        sensitivity_analysis = self.sensitivity_analysis(
            current_bookings, no_show_probabilities, ticket_prices
        )
        
        # Calculate key metrics
        baseline_revenue = strategy_comparison['baseline']['expected_revenue']
        best_strategy = max(strategy_comparison.keys(), 
                          key=lambda x: strategy_comparison[x]['expected_revenue'])
        best_revenue = strategy_comparison[best_strategy]['expected_revenue']
        improvement = best_revenue - baseline_revenue
        
        report = f"""
        OVERBOOKING OPTIMIZATION REPORT
        ==============================
        
        FLIGHT PARAMETERS:
        - Capacity: {self.flight_params.capacity} seats
        - Current Bookings: {current_bookings}
        - Base Fare: ${self.flight_params.base_fare:.2f}
        - Load Factor Target: {self.flight_params.load_factor_target:.1%}
        
        COST PARAMETERS:
        - Denied Boarding Cost: ${self.cost_params.denied_boarding_cost:.2f}
        - Empty Seat Cost: ${self.cost_params.empty_seat_cost:.2f}
        - Volunteer Compensation: ${self.cost_params.volunteer_compensation:.2f}
        
        STRATEGY COMPARISON:
        """
        
        for strategy, results in strategy_comparison.items():
            improvement_vs_baseline = results['expected_revenue'] - baseline_revenue
            report += f"""
        {strategy.upper()}:
        - Overbooking Level: {results['overbooking_level']} seats
        - Expected Revenue: ${results['expected_revenue']:,.2f}
        - Improvement vs Baseline: ${improvement_vs_baseline:+,.2f}
        - Denied Boarding Probability: {results['denied_boarding_probability']:.2%}
        - Load Factor: {results['load_factor']:.1%}
        """
        
        report += f"""
        
        RECOMMENDATIONS:
        - Best Strategy: {best_strategy.upper()}
        - Recommended Overbooking: {strategy_comparison[best_strategy]['overbooking_level']} seats
        - Expected Revenue Gain: ${improvement:,.2f} ({improvement/baseline_revenue:.1%} increase)
        - Risk Level: {strategy_comparison[best_strategy]['denied_boarding_probability']:.2%} chance of denied boarding
        
        RISK ASSESSMENT:
        - Conservative approach for risk-averse operations
        - Moderate approach for balanced risk/reward
        - Aggressive approach for revenue maximization
        - Monitor and adjust based on actual performance
        """
        
        return report
    
    def plot_optimization_results(self, optimization_result: Dict[str, Any]) -> None:
        """Create visualizations for optimization results."""
        
        results = optimization_result['all_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Revenue vs Overbooking Level
        overbooking_levels = [r['overbooking_level'] for r in results]
        revenues = [r['expected_revenue'] for r in results]
        
        axes[0, 0].plot(overbooking_levels, revenues, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].axvline(x=optimization_result['optimal_overbooking_level'], 
                          color='red', linestyle='--', label='Optimal')
        axes[0, 0].set_xlabel('Overbooking Level')
        axes[0, 0].set_ylabel('Expected Revenue ($)')
        axes[0, 0].set_title('Revenue vs Overbooking Level')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Denied Boarding Probability
        denied_probs = [r['denied_boarding_probability'] for r in results]
        
        axes[0, 1].plot(overbooking_levels, denied_probs, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].axvline(x=optimization_result['optimal_overbooking_level'], 
                          color='red', linestyle='--', label='Optimal')
        axes[0, 1].set_xlabel('Overbooking Level')
        axes[0, 1].set_ylabel('Denied Boarding Probability')
        axes[0, 1].set_title('Risk vs Overbooking Level')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Load Factor
        load_factors = [r['load_factor'] for r in results]
        
        axes[1, 0].plot(overbooking_levels, load_factors, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=self.flight_params.load_factor_target, 
                          color='orange', linestyle='--', label='Target')
        axes[1, 0].axvline(x=optimization_result['optimal_overbooking_level'], 
                          color='red', linestyle='--', label='Optimal')
        axes[1, 0].set_xlabel('Overbooking Level')
        axes[1, 0].set_ylabel('Load Factor')
        axes[1, 0].set_title('Load Factor vs Overbooking Level')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Revenue Distribution at Optimal Level
        optimal_idx = optimization_result['optimal_overbooking_level']
        if optimal_idx < len(results):
            revenue_range = np.linspace(
                results[optimal_idx]['revenue_5th_percentile'],
                results[optimal_idx]['revenue_95th_percentile'],
                100
            )
            
            axes[1, 1].hist(revenue_range, bins=20, alpha=0.7, color='skyblue', 
                           label='Revenue Distribution')
            axes[1, 1].axvline(x=results[optimal_idx]['expected_revenue'], 
                              color='red', linestyle='--', label='Expected')
            axes[1, 1].set_xlabel('Revenue ($)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Revenue Distribution (Optimal Level)')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_strategy_comparison(self, strategy_results: Dict[str, Dict[str, Any]]) -> None:
        """Plot comparison of different strategies."""
        
        strategies = list(strategy_results.keys())
        revenues = [strategy_results[s]['expected_revenue'] for s in strategies]
        risks = [strategy_results[s]['denied_boarding_probability'] for s in strategies]
        overbooking_levels = [strategy_results[s]['overbooking_level'] for s in strategies]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Revenue comparison
        colors = ['gray', 'green', 'orange', 'red', 'blue']
        bars1 = axes[0].bar(strategies, revenues, color=colors[:len(strategies)])
        axes[0].set_title('Expected Revenue by Strategy')
        axes[0].set_ylabel('Expected Revenue ($)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, revenue in zip(bars1, revenues):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(revenues)*0.01,
                        f'${revenue:,.0f}', ha='center', va='bottom')
        
        # 2. Risk vs Revenue scatter
        axes[1].scatter(risks, revenues, s=100, c=colors[:len(strategies)], alpha=0.7)
        for i, strategy in enumerate(strategies):
            axes[1].annotate(strategy, (risks[i], revenues[i]), 
                           xytext=(5, 5), textcoords='offset points')
        axes[1].set_xlabel('Denied Boarding Probability')
        axes[1].set_ylabel('Expected Revenue ($)')
        axes[1].set_title('Risk vs Revenue Trade-off')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Overbooking levels
        bars3 = axes[2].bar(strategies, overbooking_levels, color=colors[:len(strategies)])
        axes[2].set_title('Overbooking Levels by Strategy')
        axes[2].set_ylabel('Additional Seats Sold')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, level in zip(bars3, overbooking_levels):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(overbooking_levels)*0.01,
                        f'{level}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def main():
    """Demonstrate the overbooking optimization system."""
    
    print("=== AIRLINE OVERBOOKING OPTIMIZATION SYSTEM ===\n")
    
    # Load data and predictions
    df = pd.read_csv('/Users/nathanaeljohnson/opt/program/FP/airline-no-show-optimizer/data/processed/airline_data_enhanced.csv')
    
    # Simulate a specific flight scenario
    flight_sample = df.sample(150, random_state=42)  # 150 current bookings
    no_show_probabilities = flight_sample['historical_no_show_rate'].fillna(0.08).values
    ticket_prices = flight_sample['ticket_price'].values
    
    print(f"Flight Scenario:")
    print(f"- Current Bookings: {len(flight_sample)}")
    print(f"- Average Ticket Price: ${ticket_prices.mean():.2f}")
    print(f"- Average No-Show Probability: {no_show_probabilities.mean():.2%}")
    print(f"- Expected No-Shows: {np.sum(no_show_probabilities):.1f}")
    
    # Initialize optimizer with different cost scenarios
    cost_scenarios = {
        'conservative': CostParameters(
            denied_boarding_cost=1500, 
            empty_seat_cost=300,
            volunteer_compensation=1000
        ),
        'standard': CostParameters(
            denied_boarding_cost=1200, 
            empty_seat_cost=400,
            volunteer_compensation=800
        ),
        'aggressive': CostParameters(
            denied_boarding_cost=1000, 
            empty_seat_cost=500,
            volunteer_compensation=600
        )
    }
    
    flight_params = FlightParameters(capacity=180, base_fare=400, load_factor_target=0.95)
    
    print(f"\n{'='*60}")
    print("COST SCENARIO ANALYSIS")
    print(f"{'='*60}")
    
    scenario_results = {}
    for scenario_name, cost_params in cost_scenarios.items():
        print(f"\n--- {scenario_name.upper()} COST SCENARIO ---")
        
        optimizer = OverbookingOptimizer(cost_params, flight_params)
        
        # Run optimization
        optimization_result = optimizer.optimize_overbooking_level(
            len(flight_sample), no_show_probabilities, ticket_prices
        )
        
        # Compare strategies
        strategy_comparison = optimizer.compare_strategies(
            len(flight_sample), no_show_probabilities, ticket_prices
        )
        
        scenario_results[scenario_name] = {
            'optimization': optimization_result,
            'strategies': strategy_comparison
        }
        
        print(f"Optimal Overbooking: {optimization_result['optimal_overbooking_level']} seats")
        print(f"Revenue Improvement: ${optimization_result['revenue_improvement']:,.2f}")
        print(f"Denied Boarding Risk: {optimization_result['denied_boarding_probability']:.2%}")
        print(f"Load Factor: {optimization_result['load_factor']:.1%}")
    
    # Detailed analysis for standard scenario
    print(f"\n{'='*60}")
    print("DETAILED ANALYSIS - STANDARD SCENARIO")
    print(f"{'='*60}")
    
    optimizer = OverbookingOptimizer(cost_scenarios['standard'], flight_params)
    
    # Strategy comparison
    strategy_results = optimizer.compare_strategies(
        len(flight_sample), no_show_probabilities, ticket_prices
    )
    
    print("\nStrategy Comparison:")
    comparison_df = pd.DataFrame(strategy_results).T
    print(comparison_df.round(2))
    
    # Sensitivity analysis
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    sensitivity_results = optimizer.sensitivity_analysis(
        len(flight_sample), no_show_probabilities, ticket_prices
    )
    
    print("\nSensitivity to Cost Parameters:")
    for param_name, param_results in sensitivity_results['sensitivity_results'].items():
        print(f"\n{param_name.replace('_', ' ').title()}:")
        for result in param_results:
            print(f"  ${result['parameter_value']}: "
                  f"Overbooking={result['optimal_overbooking_level']}, "
                  f"Revenue=${result['expected_revenue']:,.0f}")
    
    # Generate comprehensive report
    print(f"\n{'='*60}")
    print("OPTIMIZATION REPORT")
    print(f"{'='*60}")
    
    report = optimizer.generate_optimization_report(
        len(flight_sample), no_show_probabilities, ticket_prices
    )
    print(report)
    
    # Save results
    results_summary = pd.DataFrame({
        'Scenario': list(scenario_results.keys()),
        'Optimal_Overbooking': [r['optimization']['optimal_overbooking_level'] for r in scenario_results.values()],
        'Revenue_Improvement': [r['optimization']['revenue_improvement'] for r in scenario_results.values()],
        'Denied_Boarding_Risk': [r['optimization']['denied_boarding_probability'] for r in scenario_results.values()],
        'Load_Factor': [r['optimization']['load_factor'] for r in scenario_results.values()]
    })
    
    results_summary.to_csv('/Users/nathanaeljohnson/opt/program/FP/airline-no-show-optimizer/results/overbooking_optimization.csv', index=False)
    
    print(f"\n📁 Results saved to: /results/overbooking_optimization.csv")
    
    return optimizer, scenario_results


if __name__ == "__main__":
    optimizer, results = main()