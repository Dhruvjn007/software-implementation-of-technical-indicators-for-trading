import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class MACDCalculator:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # EMA smoothing factors
        self.fast_alpha = 2 / (fast_period + 1)
        self.slow_alpha = 2 / (slow_period + 1)
        self.signal_alpha = 2 / (signal_period + 1)
        
        # State variables
        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self.prev_macd = None
        self.prev_histogram = None
        
        # Data storage for plotting
        self.prices = []
        self.macd_values = []
        self.signal_values = []
        self.histogram_values = []
        self.signals = []
        
    def update(self, price):
        """Process one price point at a time - simulating real-time data"""
        self.prices.append(price)
        
        # Calculate fast EMA
        if self.fast_ema is None:
            self.fast_ema = price
        else:
            self.fast_ema = (price * self.fast_alpha) + (self.fast_ema * (1 - self.fast_alpha))
        
        # Calculate slow EMA
        if self.slow_ema is None:
            self.slow_ema = price
        else:
            self.slow_ema = (price * self.slow_alpha) + (self.slow_ema * (1 - self.slow_alpha))
        
        # Calculate MACD line
        macd = self.fast_ema - self.slow_ema
        self.macd_values.append(macd)
        
        # Calculate signal line (EMA of MACD)
        if self.signal_ema is None:
            self.signal_ema = macd
        else:
            self.signal_ema = (macd * self.signal_alpha) + (self.signal_ema * (1 - self.signal_alpha))
        
        self.signal_values.append(self.signal_ema)
        
        # Calculate histogram
        histogram = macd - self.signal_ema
        self.histogram_values.append(histogram)
        
        # Generate trading signal
        signal = self._generate_signal(macd, histogram)
        self.signals.append(signal)
        
        # Update previous values for next iteration
        self.prev_macd = macd
        self.prev_histogram = histogram
        
        return macd, self.signal_ema, histogram, signal
    
    def _generate_signal(self, macd, histogram):
        """Generate buy/sell/hold signals based on MACD crossover and histogram"""
        # Need at least 2 data points to detect crossover
        if self.prev_macd is None or self.prev_histogram is None:
            return 0  # Hold
        
        # MACD line crosses above signal line (bullish)
        if self.prev_histogram <= 0 and histogram > 0:
            return 1  # Buy
        
        # MACD line crosses below signal line (bearish)
        elif self.prev_histogram >= 0 and histogram < 0:
            return -1  # Sell
        
        return 0  # Hold

def generate_realistic_prices(n_points, initial_price=100, volatility=0.02):
    """Generate realistic price series using geometric brownian motion"""
    np.random.seed(42)  # For reproducible results
    
    prices = [initial_price]
    
    for _ in range(n_points - 1):
        # Random walk with drift and volatility
        drift = 0.0001  # Small positive drift
        shock = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + drift + shock)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, 0.01)
        prices.append(new_price)
    
    return prices

def run_macd_simulation():
    """Run the complete MACD trading simulation"""
    # Generate 1 million price points
    print("Generating 1 million price points...")
    prices = generate_realistic_prices(1000000)

    start=time.time() 
    # Initialize MACD calculator
    macd_calc = MACDCalculator()
    
    print("Processing prices sequentially (simulating real-time trading)...")
    
    # Process each price one by one (simulating real-time)
    for i, price in enumerate(prices):
        macd_calc.update(price)
        
        # Progress indicator for large dataset
        if (i + 1) % 100000 == 0:
            print(f"Processed {i + 1:,} prices")

    end=time.time()
    print(f"Total processing time: {end-start:.2f} seconds")

    print("Simulation complete! Creating plots...")
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Sample data for plotting (last 5000 points for clarity)
    sample_size = 5000
    start_idx = -sample_size
    
    sample_prices = macd_calc.prices[start_idx:]
    sample_macd = macd_calc.macd_values[start_idx:]
    sample_signal = macd_calc.signal_values[start_idx:]
    sample_histogram = macd_calc.histogram_values[start_idx:]
    sample_signals = macd_calc.signals[start_idx:]
    
    x_axis = range(len(sample_prices))
    
    # Plot 1: Price chart
    axes[0].plot(x_axis, sample_prices, 'b-', linewidth=0.8, alpha=0.7)
    axes[0].set_title('Price Chart (Last 5000 Points)', fontsize=12)
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: MACD and Signal lines
    axes[1].plot(x_axis, sample_macd, 'b-', label='MACD Line', linewidth=0.8)
    axes[1].plot(x_axis, sample_signal, 'r-', label='Signal Line', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title('MACD and Signal Lines', fontsize=12)
    axes[1].set_ylabel('MACD Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram
    colors = ['red' if h < 0 else 'green' for h in sample_histogram]
    axes[2].bar(x_axis, sample_histogram, color=colors, alpha=0.6, width=1)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.8)
    axes[2].set_title('MACD Histogram', fontsize=12)
    axes[2].set_ylabel('Histogram')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Trading signals
    buy_signals = [i for i, s in enumerate(sample_signals) if s == 1]
    sell_signals = [i for i, s in enumerate(sample_signals) if s == -1]
    
    axes[3].plot(x_axis, sample_prices, 'b-', alpha=0.5, linewidth=0.8)
    if buy_signals:
        buy_prices = [sample_prices[i] for i in buy_signals]
        axes[3].scatter(buy_signals, buy_prices, color='green', marker='^', 
                       s=50, alpha=0.8, label='Buy Signal')
    if sell_signals:
        sell_prices = [sample_prices[i] for i in sell_signals]
        axes[3].scatter(sell_signals, sell_prices, color='red', marker='v', 
                       s=50, alpha=0.8, label='Sell Signal')
    
    axes[3].set_title('Trading Signals on Price Chart', fontsize=12)
    axes[3].set_xlabel('Time Index (Last 5000 Points)')
    axes[3].set_ylabel('Price')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    total_signals = len(macd_calc.signals)
    buy_count = sum(1 for s in macd_calc.signals if s == 1)
    sell_count = sum(1 for s in macd_calc.signals if s == -1)
    hold_count = total_signals - buy_count - sell_count
    
    print(f"\nSimulation Summary:")
    print(f"Total data points processed: {total_signals:,}")
    print(f"Buy signals: {buy_count:,} ({buy_count/total_signals*100:.2f}%)")
    print(f"Sell signals: {sell_count:,} ({sell_count/total_signals*100:.2f}%)")
    print(f"Hold signals: {hold_count:,} ({hold_count/total_signals*100:.2f}%)")

if __name__ == "__main__":
    run_macd_simulation()