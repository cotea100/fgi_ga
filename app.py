import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib
import os
import random
import time
from datetime import datetime
from fpdf import FPDF

# Configure matplotlib
matplotlib.use("Agg")
st.set_page_config(layout="wide", page_title="FGI Strategy Optimization")

# File upload function for Streamlit
def upload_file():
    """Get file upload from user using Streamlit file uploader."""
    st.title("FGI Strategy Optimization with Genetic Algorithm")
    st.write("This app helps you find optimal Fear & Greed Index (FGI) trading strategies using genetic algorithms.")
    uploaded_file = st.file_uploader("Upload Excel or CSV file containing date, price, and FGI data", 
                                   type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.info("Please upload a file to continue")
        st.stop()
    
    st.success(f"Uploaded: {uploaded_file.name}")
    return uploaded_file.name, {"file": uploaded_file}

# Read uploaded file function
def read_uploaded_file(uploaded):
    """Read the uploaded file into a pandas DataFrame."""
    file = uploaded["file"]
    
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xls") or file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Only CSV, XLS, XLSX files are supported.")

# Backtesting function (FGI strategy)
def backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital=10_000_000, commission_rate=0.0025):
    """
    Perform FGI strategy backtesting.

    Buy condition: FGI was < buy_threshold and then rises to > buy_threshold
    Sell condition: FGI was > sell_threshold and then falls to < sell_threshold
    """
    cash = initial_capital
    units = 0
    current_state = 'CASH'
    portfolio = []
    dates = []
    trade_count = 0
    trade_history = []  # For visualizing trades

    # Initial flags
    fgi_low_seen_since_sell = False  # Have we seen FGI < buy_threshold since last sell
    fgi_high_seen_since_buy = False  # Have we seen FGI > sell_threshold since last buy

    # Need at least 2 days of data for simulation
    if len(df) < 2:
        st.warning("Data is too short. At least 2 days of data are required.")
        return {
            "Strategy": f"Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}",
            "ROI": 0,
            "CAGR": 0,
            "MDD": 0,
            "Calmar": 0,
            "Trades": 0,
            "Final Value": initial_capital,
            "portfolio_values": [initial_capital],
            "trade_history": []
        }

    # Force buy on first day (decision on day 0, execution on day 1)
    pending_action = ('Buy', 0)

    # Main simulation loop
    for i in range(len(df)):
        date_i = df['date'].iloc[i]
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]

        # Check if next day exists
        has_next_day = (i < len(df) - 1)

        # Execute decision made yesterday (day i-1) today (day i)
        if pending_action is not None and pending_action[1] == i - 1:
            action_type = pending_action[0]

            if action_type == 'Buy':
                if not pd.isna(price_i) and price_i > 0:
                    # Calculate buy amount considering commission
                    buy_amount = cash / (1 + commission_rate)
                    units = buy_amount / price_i
                    cash = 0
                    current_state = 'STOCK'
                    trade_count += 1
                    
                    # Record trade for visualization
                    trade_history.append({
                        'date': date_i,
                        'action': 'Buy',
                        'price': price_i,
                        'fgi': fgi_i
                    })

                    # Reset FGI flags after buy
                    fgi_low_seen_since_sell = False
                    fgi_high_seen_since_buy = False

            elif action_type == 'Sell':
                if not pd.isna(price_i) and price_i > 0:
                    # Calculate sell amount considering commission
                    sell_amount = units * price_i
                    cash = sell_amount * (1 - commission_rate)
                    units = 0
                    current_state = 'CASH'
                    trade_count += 1
                    
                    # Record trade for visualization
                    trade_history.append({
                        'date': date_i,
                        'action': 'Sell',
                        'price': price_i,
                        'fgi': fgi_i
                    })

                    # Reset FGI flags after sell
                    fgi_low_seen_since_sell = False
                    fgi_high_seen_since_buy = False

            # Clear pending action after execution
            pending_action = None

        # Make decision for tomorrow (day i+1) based on today's FGI (day i)
        if has_next_day and pending_action is None and not pd.isna(fgi_i):
            if current_state == 'CASH':
                # Buy strategy logic
                if fgi_i < buy_threshold:
                    fgi_low_seen_since_sell = True

                if fgi_low_seen_since_sell and fgi_i > buy_threshold:
                    pending_action = ('Buy', i)

            elif current_state == 'STOCK':
                # Sell strategy logic
                if fgi_i > sell_threshold:
                    fgi_high_seen_since_buy = True

                if fgi_high_seen_since_buy and fgi_i < sell_threshold:
                    pending_action = ('Sell', i)

        # Calculate and record portfolio value
        if current_state == 'CASH':
            portfolio_value = cash
        else:  # 'STOCK'
            if pd.isna(price_i) or price_i <= 0:
                portfolio_value = np.nan
            else:
                portfolio_value = units * price_i

        portfolio.append(portfolio_value)
        dates.append(date_i)

    # Calculate results
    series = pd.Series(portfolio, index=dates)
    valid_series = series.dropna()

    if len(valid_series) <= 1:
        return {
            "Strategy": f"Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}",
            "ROI": 0,
            "CAGR": 0,
            "MDD": 0,
            "Calmar": 0,
            "Trades": trade_count,
            "Final Value": initial_capital,
            "portfolio_values": portfolio,
            "trade_history": trade_history
        }

    # Final portfolio value
    final_portfolio_value = valid_series.iloc[-1]

    # ROI calculation
    roi = (final_portfolio_value / initial_capital) - 1

    # Period (years) calculation
    start_date = valid_series.index[0]
    end_date = valid_series.index[-1]
    n_years = (end_date - start_date).days / 365.25
    if n_years <= 0:
        n_years = 1/365  # Very short period exception handling

    # CAGR calculation
    cagr = (final_portfolio_value / initial_capital) ** (1 / n_years) - 1

    # MDD calculation
    mdd = ((valid_series - valid_series.cummax()) / valid_series.cummax()).min()

    # Calmar ratio calculation
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    return {
        "Strategy": f"Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}",
        "ROI": roi,
        "CAGR": cagr,
        "MDD": mdd,
        "Calmar": calmar,
        "Trades": trade_count,
        "Final Value": final_portfolio_value,
        "portfolio_values": portfolio,
        "dates": dates,
        "trade_history": trade_history
    }

# Enhanced fitness calculation function with multiple objectives
def calculate_multi_fitness(individual, df, initial_capital, commission_rate, weights=None):
    """
    Calculate multiple fitness metrics for an individual.
    
    Parameters:
    -----------
    individual : tuple
        (buy_threshold, sell_threshold) tuple representing an individual
    df : pandas DataFrame
        DataFrame containing date, price, and fgi columns
    initial_capital : float
        Initial investment amount
    commission_rate : float
        Trading commission rate (e.g., 0.0025 for 0.25%)
    weights : dict, optional
        Dictionary of weights for different metrics. Default weights are:
        {
            'roi': 0.5,       # Return on Investment
            'mdd': 0.2,       # Maximum Drawdown (negative)
            'calmar': 0.15,   # Calmar Ratio
            'trades': 0.05,   # Trade frequency optimization
            'volatility': 0.1 # Volatility of returns
        }
        
    Returns:
    --------
    dict
        Dictionary containing individual fitness metrics and composite score
    """
    buy_threshold, sell_threshold = individual
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'roi': 0.5,       # Return on Investment (higher is better)
            'mdd': 0.2,       # Maximum Drawdown (lower is better)
            'calmar': 0.15,   # Calmar Ratio (higher is better)
            'trades': 0.05,   # Trade frequency optimization (optimal range is best)
            'volatility': 0.1 # Volatility of returns (lower is better)
        }
    
    # Ensure weights sum to 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:  # Allow small tolerance
        # Normalize weights
        for key in weights:
            weights[key] = weights[key] / weight_sum
    
    # Run backtesting
    result = backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital, commission_rate)
    
    # Extract basic metrics from backtest result
    roi = result['ROI']
    mdd = result['MDD']
    calmar = result['Calmar']
    trades = result['Trades']
    
    # Calculate additional metrics that might not be in the original backtest result
    
    # 1. Calculate return volatility if we have portfolio values over time
    if 'portfolio_values' in result:
        portfolio_values = result['portfolio_values']
        # Convert to returns
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0.0
    else:
        # If portfolio values not available, estimate volatility from MDD
        # This is an approximation: higher MDD often correlates with higher volatility
        volatility = abs(mdd) / 2.0  # Rough approximation
    
    # 2. Calculate trades metric - penalize too few or too many trades
    # Optimal trading frequency depends on the strategy and time period
    # Here we define an optimal range and penalize deviations
    if trades < 3:  # Too few trades
        trades_score = 0.3
    elif trades < 8:  # Below optimal
        trades_score = 0.7
    elif trades <= 20:  # Optimal range
        trades_score = 1.0
    elif trades <= 50:  # Above optimal but still reasonable
        trades_score = 0.8
    else:  # Too many trades
        trades_score = 0.5
    
    # Calculate individual fitness components (normalize where needed)
    # Each component should be designed so that higher is better
    roi_fitness = roi
    mdd_fitness = -abs(mdd)  # Negative because we want to minimize drawdown
    calmar_fitness = min(calmar, 10.0) / 10.0  # Normalize and cap extremely high values
    trades_fitness = trades_score
    volatility_fitness = -min(volatility, 0.5) / 0.5  # Negative & normalized (higher is better)
    
    # Store all individual fitness metrics
    fitness_metrics = {
        'roi': roi_fitness,
        'mdd': mdd_fitness,
        'calmar': calmar_fitness,
        'trades': trades_fitness,
        'volatility': volatility_fitness,
    }
    
    # Calculate composite fitness using weights
    composite_fitness = 0.0
    for metric, value in fitness_metrics.items():
        if metric in weights:
            composite_fitness += weights[metric] * value
    
    # Create and return comprehensive fitness result
    fitness_result = {
        'individual': individual,
        'metrics': {
            'roi': roi,
            'mdd': mdd,
            'calmar': calmar,
            'trades': trades,
            'volatility': volatility if 'portfolio_values' in result else 'estimated',
        },
        'fitness_components': fitness_metrics,
        'weights': weights,
        'composite': composite_fitness
    }
    
    return fitness_result

# Function to select elite individuals based on different fitness metrics
def select_diverse_elite(population, fitness_results, num_elite=5):
    """Select elite individuals that excel in different metrics."""
    if not fitness_results or len(fitness_results) < num_elite:
        return population[:min(num_elite, len(population))]
    
    # Create a dictionary mapping individuals to their fitness results
    fitness_dict = {tuple(result['individual']): result for result in fitness_results}
    
    elite = []
    
    # Select best individual by composite score
    best_composite = max(fitness_results, key=lambda x: x['composite'])
    elite.append(tuple(best_composite['individual']))
    
    # If we need more elite individuals, select best by different metrics
    if num_elite > 1:
        metrics = ['roi', 'mdd', 'calmar', 'trades', 'volatility']
        
        for metric in metrics:
            if len(elite) >= num_elite:
                break
                
            # For MDD and volatility, we want the minimum (but fitness is negative)
            if metric in ['mdd', 'volatility']:
                best = max(fitness_results, key=lambda x: x['fitness_components'][metric])
            else:
                best = max(fitness_results, key=lambda x: x['fitness_components'][metric])
                
            best_individual = tuple(best['individual'])
            if best_individual not in elite:
                elite.append(best_individual)
                
    # If we still need more, pick random diverse individuals
    remaining_population = [ind for ind in population if tuple(ind) not in elite]
    while len(elite) < num_elite and remaining_population:
        # Pick a random individual
        random_idx = random.randint(0, len(remaining_population) - 1)
        elite.append(tuple(remaining_population.pop(random_idx)))
    
    return elite

# Modified selection function for multi-objective optimization
def selection_multi_objective(population, fitness_results, num_parents):
    """Select parents using a combination of methods for multi-objective optimization."""
    if not fitness_results:
        return random.sample(population, min(num_parents, len(population)))
    
    # Create a mapping from individuals to fitness results
    fitness_dict = {tuple(result['individual']): result for result in fitness_results}
    
    # 1. First, select some elite individuals (20% of parents)
    num_elite = max(1, int(num_parents * 0.2))
    elite = select_diverse_elite(population, fitness_results, num_elite)
    parents = list(elite)
    
    # 2. Select the rest using tournament selection with multiple objectives
    remaining_slots = num_parents - len(parents)
    if remaining_slots > 0:
        for _ in range(remaining_slots):
            # Tournament size (about 15% of population size)
            tournament_size = max(3, int(len(population) * 0.15))
            
            # Random tournament selection
            tournament = random.sample(population, tournament_size)
            
            # Calculate tournament scores
            tournament_scores = []
            for individual in tournament:
                if tuple(individual) in fitness_dict:
                    # Use composite fitness score for tournament
                    score = fitness_dict[tuple(individual)]['composite']
                    tournament_scores.append((individual, score))
                else:
                    # If fitness not found, assign a low score
                    tournament_scores.append((individual, -1.0))
            
            # Select winner (highest score)
            winner, _ = max(tournament_scores, key=lambda x: x[1])
            parents.append(tuple(winner))
    
    return parents

# Initial population creation function
def create_initial_population(population_size, buy_threshold_range, sell_threshold_range):
    """Create initial population of strategies."""
    population = []
    
    for _ in range(population_size):
        buy_threshold = random.randint(buy_threshold_range[0], buy_threshold_range[1])
        sell_threshold = random.randint(sell_threshold_range[0], sell_threshold_range[1])
        population.append((buy_threshold, sell_threshold))
    
    return population

# Crossover function
def crossover(parents, offspring_size):
    """Create offspring from parents using uniform crossover."""
    offspring = []
    
    while len(offspring) < offspring_size:
        # Randomly select two parents
        parent1, parent2 = random.sample(parents, 2)
        
        # Crossover per parameter
        buy_threshold = parent1[0] if random.random() < 0.5 else parent2[0]
        sell_threshold = parent1[1] if random.random() < 0.5 else parent2[1]
        
        offspring.append((buy_threshold, sell_threshold))
    
    return offspring

# Mutation function
def mutation(offspring, mutation_rate, buy_threshold_range, sell_threshold_range):
    """Apply mutation to offspring."""
    mutated_offspring = []
    
    for individual in offspring:
        buy_threshold, sell_threshold = individual
        
        # Buy threshold mutation
        if random.random() < mutation_rate:
            # Change between -3 and +3 from current value
            change = random.randint(-3, 3)
            buy_threshold = max(buy_threshold_range[0], min(buy_threshold + change, buy_threshold_range[1]))
        
        # Sell threshold mutation
        if random.random() < mutation_rate:
            change = random.randint(-3, 3)
            sell_threshold = max(sell_threshold_range[0], min(sell_threshold + change, sell_threshold_range[1]))
        
        mutated_offspring.append((buy_threshold, sell_threshold))
    
    return mutated_offspring

# Function to get detailed analysis of best strategies
def analyze_best_strategies(fitness_results, n=5):
    """Analyze the top N strategies based on different metrics."""
    if not fitness_results or len(fitness_results) == 0:
        return {}
    
    # Create deep copies to avoid modifying original data
    results_copy = fitness_results.copy()
    
    # Find best strategies by different metrics
    best_by = {
        'composite': sorted(results_copy, key=lambda x: x['composite'], reverse=True)[:n],
        'roi': sorted(results_copy, key=lambda x: x['metrics']['roi'], reverse=True)[:n],
        'mdd': sorted(results_copy, key=lambda x: x['metrics']['mdd'])[:n],  # Lower MDD is better
        'calmar': sorted(results_copy, key=lambda x: x['metrics']['calmar'], reverse=True)[:n],
        'trades': sorted(results_copy, key=lambda x: abs(x['metrics']['trades'] - 15))[:n]  # Closest to optimal trade count
    }
    
    return best_by

# Plot progress using Streamlit
def plot_ga_progress(best_fitness_history, avg_fitness_history, buy_thresh_history, sell_thresh_history, generation_count):
    """Visualize genetic algorithm progress using Streamlit."""
    st.subheader("Genetic Algorithm Progress")
    
    col1, col2 = st.columns(2)
    
    # 1. Best and Average Fitness chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(1, generation_count + 1), best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(range(1, generation_count + 1), avg_fitness_history, 'r-', linewidth=2, label='Average Fitness')
    ax1.set_title('Fitness Trend')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.grid(True)
    ax1.legend()
    col1.pyplot(fig1)
    
    # 2. Threshold Trends
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(range(1, generation_count + 1), buy_thresh_history, 'g-', linewidth=2, label='Buy Threshold')
    ax2.plot(range(1, generation_count + 1), sell_thresh_history, 'm-', linewidth=2, label='Sell Threshold')
    ax2.set_title('Threshold Trends')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Threshold Values')
    ax2.grid(True)
    ax2.legend()
    col2.pyplot(fig2)

# Plot strategy results using Streamlit
def plot_strategy_results(df, buy_threshold, sell_threshold, initial_capital, result):
    """Visualize strategy backtesting results using Streamlit."""
    st.subheader(f"Strategy Results: Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    # 1. FGI and price chart with trade markers
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # Price on left y-axis
    ax1.plot(df['date'], df['price'], 'b-', linewidth=1.5, label='Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # FGI on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['fgi'], 'r-', linewidth=1, label='FGI', alpha=0.7)
    ax2.axhline(y=buy_threshold, color='g', linestyle='--', label=f'Buy Threshold ({buy_threshold})')
    ax2.axhline(y=sell_threshold, color='m', linestyle='--', label=f'Sell Threshold ({sell_threshold})')
    ax2.set_ylabel('FGI', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Mark trades on the chart
    trade_history = result.get('trade_history', [])
    buy_dates = [trade['date'] for trade in trade_history if trade['action'] == 'Buy']
    buy_prices = [trade['price'] for trade in trade_history if trade['action'] == 'Buy']
    
    sell_dates = [trade['date'] for trade in trade_history if trade['action'] == 'Sell']
    sell_prices = [trade['price'] for trade in trade_history if trade['action'] == 'Sell']
    
    ax1.scatter(buy_dates, buy_prices, color='g', marker='^', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_dates, sell_prices, color='r', marker='v', s=100, label='Sell', zorder=5)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Price, FGI, and Trades')
    plt.grid(True, alpha=0.3)
    col1.pyplot(fig1)
    
    # 2. Portfolio value over time
    if 'portfolio_values' in result and 'dates' in result:
        fig2, ax3 = plt.subplots(figsize=(12, 7))
        portfolio_values = result['portfolio_values']
        dates = result['dates']
        
        # Calculate buy & hold for comparison
        buy_and_hold_values = (df['price'] / df['price'].iloc[0]) * initial_capital
        
        # Plot portfolio value
        ax3.plot(dates, portfolio_values, 'b-', linewidth=2, label='Strategy Portfolio')
        ax3.plot(df['date'], buy_and_hold_values, 'g--', linewidth=1.5, label='Buy & Hold')
        
        # Mark trades on portfolio chart too
        for trade in trade_history:
            if trade['action'] == 'Buy':
                ax3.axvline(x=trade['date'], color='g', linestyle=':', alpha=0.5)
            else:  # 'Sell'
                ax3.axvline(x=trade['date'], color='r', linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Portfolio Value')
        ax3.set_title('Portfolio Value Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        col2.pyplot(fig2)
    
    # Performance metrics
    metrics_cols = st.columns(5)
    
    metrics_cols[0].metric("ROI", f"{result['ROI']:.2%}")
    metrics_cols[1].metric("CAGR", f"{result['CAGR']:.2%}")
    metrics_cols[2].metric("MDD", f"{result['MDD']:.2%}")
    metrics_cols[3].metric("Calmar Ratio", f"{result['Calmar']:.2f}")
    metrics_cols[4].metric("Trades", f"{result['Trades']}")
    
    # Buy & Hold comparison
    bh_roi = (buy_and_hold_values.iloc[-1] / initial_capital) - 1 if len(buy_and_hold_values) > 0 else 0
    st.metric("Strategy vs Buy & Hold", f"{result['ROI'] - bh_roi:.2%}", 
             f"{result['ROI']:.2%} vs {bh_roi:.2%}")

@st.cache_data
def run_genetic_algorithm(df, params):
    """Run genetic algorithm with caching for performance."""
    # Extract parameters
    initial_capital = params['initial_capital']
    commission_rate = params['commission_rate']
    population_size = params['population_size']
    num_generations = params['num_generations']
    mutation_rate = params['mutation_rate']
    num_parents = params['num_parents']
    buy_threshold_range = params['buy_threshold_range']
    sell_threshold_range = params['sell_threshold_range']
    fitness_weights = params['fitness_weights']
    
    # Setup for tracking
    best_fitness_history = []
    avg_fitness_history = []
    best_individual_history = []
    buy_thresh_history = []
    sell_thresh_history = []
    all_results = []
    all_fitness_results = []
    
    # Create status placeholder
    status = st.empty()
    progress_bar = st.progress(0)
    
    # Create initial population
    population = create_initial_population(population_size, buy_threshold_range, sell_threshold_range)
    
    # Genetic algorithm generation loop
    start_time = time.time()
    for generation in range(1, num_generations + 1):
        generation_start = time.time()
        
        # Update status
        status.text(f"Running generation {generation}/{num_generations}...")
        progress_bar.progress(generation / num_generations)
        
        # Calculate fitness for population
        fitness_results = [
            calculate_multi_fitness(individual, df, initial_capital, commission_rate, fitness_weights)
            for individual in population
        ]
        
        # Extract fitness values
        fitnesses = [result['composite'] for result in fitness_results]
        avg_fitness = np.mean(fitnesses)
        
        # Find best individual
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]
        best_fitness_result = fitness_results[best_idx]
        
        # Store for tracking
        all_fitness_results.append(best_fitness_result)
        
        # Backtest best individual of this generation
        buy_threshold, sell_threshold = best_individual
        best_result = backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital, commission_rate)
        all_results.append(best_result)
        
        # Record progress
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        best_individual_history.append(best_individual)
        buy_thresh_history.append(buy_threshold)
        sell_thresh_history.append(sell_threshold)
        
        # Selection
        parents = selection_multi_objective(population, fitness_results, num_parents)
        
        # Crossover
        offspring_size = population_size - len(parents)
        offspring = crossover(parents, offspring_size)