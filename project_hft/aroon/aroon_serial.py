import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time

def generate_realistic_prices(n_prices=10000, initial_price=100):
    """
    Generate realistic stock prices using geometric Brownian motion with trend changes,
    volatility clustering, and market regime switches
    """
    prices = [initial_price]
    
    # Market regime parameters
    current_regime = 'normal'  # normal, bull, bear, volatile
    regime_duration = 0
    trend = 0.0005  # daily drift
    volatility = 0.02  # daily volatility
    
    for i in range(1, n_prices):
        # Regime switching logic
        if regime_duration <= 0:
            regime_probs = {'normal': 0.7, 'bull': 0.1, 'bear': 0.1, 'volatile': 0.1}
            current_regime = np.random.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
            regime_duration = np.random.randint(50, 500)  # regime lasts 50-500 periods
        
        # Set parameters based on regime
        if current_regime == 'bull':
            trend = np.random.uniform(0.001, 0.003)
            volatility = np.random.uniform(0.015, 0.025)
        elif current_regime == 'bear':
            trend = np.random.uniform(-0.003, -0.001)
            volatility = np.random.uniform(0.02, 0.035)
        elif current_regime == 'volatile':
            trend = np.random.uniform(-0.001, 0.001)
            volatility = np.random.uniform(0.03, 0.05)
        else:  # normal
            trend = np.random.uniform(-0.0005, 0.0005)
            volatility = np.random.uniform(0.015, 0.025)
        
        # Add some mean reversion
        if i > 100:
            ma_100 = sum(prices[-100:]) / 100
            mean_reversion = (ma_100 - prices[-1]) * 0.001
            trend += mean_reversion
        
        # Generate price change using geometric Brownian motion
        random_shock = np.random.normal(0, 1)
        price_change = trend + volatility * random_shock
        
        # Apply price change
        new_price = prices[-1] * (1 + price_change)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, 0.01)
        
        prices.append(new_price)
        regime_duration -= 1
    
    return prices

def compute_aroon_and_signals(prices, lookback_period=14):
    """
    Compute Aroon oscillator and generate trading signals in real-time simulation
    """
    aroon_up_values = []
    aroon_down_values = []
    aroon_oscillator = []
    signals = []
    
    # Use deque for efficient sliding window operations
    price_window = deque(maxlen=lookback_period)
    
    for i, price in enumerate(prices):
        price_window.append(price)
        
        if len(price_window) < lookback_period:
            # Not enough data yet
            aroon_up_values.append(np.nan)
            aroon_down_values.append(np.nan)
            aroon_oscillator.append(np.nan)
            signals.append('hold')
            continue
        
        # Find highest high and lowest low in current window
        window_list = list(price_window)
        highest_high = max(window_list)
        lowest_low = min(window_list)
        
        # Find periods since highest high and lowest low
        periods_since_high = lookback_period - 1 - window_list[::-1].index(highest_high)
        periods_since_low = lookback_period - 1 - window_list[::-1].index(lowest_low)
        
        # Calculate Aroon Up and Aroon Down
        aroon_up = ((lookback_period - periods_since_high) / lookback_period) * 100
        aroon_down = ((lookback_period - periods_since_low) / lookback_period) * 100
        
        # Calculate Aroon Oscillator
        aroon_osc = aroon_up - aroon_down
        
        aroon_up_values.append(aroon_up)
        aroon_down_values.append(aroon_down)
        aroon_oscillator.append(aroon_osc)
        
        # Generate trading signals
        if len(aroon_oscillator) < 2:
            signals.append('hold')
        else:
            prev_osc = aroon_oscillator[-2]
            curr_osc = aroon_osc
            
            # Signal generation logic
            if curr_osc > 50 and prev_osc <= 50:
                signal = 'buy'
            elif curr_osc < -50 and prev_osc >= -50:
                signal = 'sell'
            elif curr_osc > 0 and aroon_up > 70:
                signal = 'buy'
            elif curr_osc < 0 and aroon_down > 70:
                signal = 'sell'
            else:
                signal = 'hold'
            
            signals.append(signal)
    
    return aroon_up_values, aroon_down_values, aroon_oscillator, signals

def plot_results(prices, aroon_oscillator, signals, sample_size=5000):
    """
    Plot the results showing prices, Aroon oscillator, and trading signals
    """
    # Use only a sample for visualization (last sample_size points)
    start_idx = max(0, len(prices) - sample_size)
    
    sample_prices = prices[start_idx:]
    sample_aroon = aroon_oscillator[start_idx:]
    sample_signals = signals[start_idx:]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot prices
    ax1.plot(sample_prices, color='black', linewidth=1, label='Price')
    ax1.set_title(f'Stock Price (Last {len(sample_prices)} periods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Aroon oscillator
    ax2.plot(sample_aroon, color='blue', linewidth=1, label='Aroon Oscillator')
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='Buy Threshold')
    ax2.axhline(y=-50, color='red', linestyle='--', alpha=0.7, label='Sell Threshold')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('Aroon Oscillator', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Aroon Value')
    ax2.set_ylim(-100, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot trading signals
    buy_points = [i for i, signal in enumerate(sample_signals) if signal == 'buy']
    sell_points = [i for i, signal in enumerate(sample_signals) if signal == 'sell']
    hold_points = [i for i, signal in enumerate(sample_signals) if signal == 'hold']
    
    if buy_points:
        ax3.scatter(buy_points, [1]*len(buy_points), color='green', marker='^', 
                   s=30, label=f'Buy ({len(buy_points)})', alpha=0.7)
    if sell_points:
        ax3.scatter(sell_points, [-1]*len(sell_points), color='red', marker='v', 
                   s=30, label=f'Sell ({len(sell_points)})', alpha=0.7)
    if hold_points:
        ax3.scatter(hold_points, [0]*len(hold_points), color='gray', marker='o', 
                   s=10, label=f'Hold ({len(hold_points)})', alpha=0.3)
    
    ax3.set_title('Trading Signals', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Signal')
    ax3.set_xlabel('Time Period')
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Sell', 'Hold', 'Buy'])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    total_signals = len(sample_signals)
    buy_count = sample_signals.count('buy')
    sell_count = sample_signals.count('sell')
    hold_count = sample_signals.count('hold')
    
    print(f"\nTrading Summary (Last {total_signals} periods):")
    print(f"Buy signals: {buy_count} ({buy_count/total_signals*100:.1f}%)")
    print(f"Sell signals: {sell_count} ({sell_count/total_signals*100:.1f}%)")
    print(f"Hold signals: {hold_count} ({hold_count/total_signals*100:.1f}%)")

def main():
    """
    Main function to run the complete trading simulation
    """
    print("Computing started...")
    
    # Generate realistic prices
    print("Generating realistic price data...")
    prices = generate_realistic_prices(n_prices=10000, initial_price=100)
    
    # Compute Aroon and signals in real-time simulation
    start=time.time()
    print("Computing Aroon oscillator and trading signals...")
    aroon_up, aroon_down, aroon_osc, signals = compute_aroon_and_signals(prices, lookback_period=14)
    end=time.time()
    print(f"Computing ended in {end-start:.2f} seconds")

    # Plot results
    print("Creating visualization...")
    plot_results(prices, aroon_osc, signals, sample_size=5000)
    
    print("Graph plotted")
    
    return prices, aroon_osc, signals

# Run the simulation
if __name__ == "__main__":
    prices, aroon_oscillator, trading_signals = main()