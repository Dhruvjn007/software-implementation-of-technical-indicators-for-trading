import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import time

class AroonPipelineProcessor:
    """
    Parallel pipeline processor for Aroon indicator following the architecture diagram
    """
    def __init__(self, lookback_period=14):
        self.lookback_period = lookback_period
        self.price_window = deque(maxlen=lookback_period)
        self.results = []
        self.lock = threading.Lock()
        
    def process_price_chunk(self, price_chunk, start_idx):
        """
        Process a chunk of prices in parallel pipeline fashion
        """
        chunk_results = []
        local_window = deque(maxlen=self.lookback_period)
        
        # Initialize local window with previous context if needed
        with self.lock:
            if len(self.price_window) > 0:
                local_window.extend(list(self.price_window)[-self.lookback_period+1:])
        
        for i, price in enumerate(price_chunk):
            local_window.append(price)
            
            if len(local_window) < self.lookback_period:
                chunk_results.append((0, 0))  # aroon_oscillator, signal
                continue
            
            # Parallel pipeline components as shown in diagram
            hp_lp_result = self.compute_high_low_prices(local_window)
            gt_lw_result = self.compute_price_comparisons(local_window, hp_lp_result)
            hc_lc_result = self.compute_high_low_counts(gt_lw_result)
            au_ad_result = self.compute_aroon_values(hc_lc_result)
            
            aroon_oscillator = au_ad_result['aroon_up'] - au_ad_result['aroon_down']
            signal = self.generate_signal(aroon_oscillator)
            
            chunk_results.append((aroon_oscillator, signal))
        
        return start_idx, chunk_results
    
    def compute_high_low_prices(self, window):
        """
        hp, lp: Compute high and low prices from window
        """
        price_list = list(window)
        return {
            'highest_high': max(price_list),
            'lowest_low': min(price_list)
        }
    
    def compute_price_comparisons(self, window, hp_lp_data):
        """
        gt, lw: Compare current and previous prices with high/low
        """
        price_list = list(window)
        comparisons = []
        
        for i, price in enumerate(price_list):
            gt_high = 1 if price == hp_lp_data['highest_high'] else 0
            lw_low = 1 if price == hp_lp_data['lowest_low'] else 0
            comparisons.append({
                'index': i,
                'is_high': gt_high,
                'is_low': lw_low
            })
        
        return comparisons
    
    def compute_high_low_counts(self, comparisons):
        """
        hc, lc: Count positions of highs and lows
        """
        periods_since_high = self.lookback_period - 1  # Default to oldest
        periods_since_low = self.lookback_period - 1   # Default to oldest
        
        # Find most recent high and low (search backwards)
        for i in range(len(comparisons) - 1, -1, -1):
            if comparisons[i]['is_high'] and periods_since_high == self.lookback_period - 1:
                periods_since_high = len(comparisons) - 1 - i
            if comparisons[i]['is_low'] and periods_since_low == self.lookback_period - 1:
                periods_since_low = len(comparisons) - 1 - i
        
        return {
            'periods_since_high': periods_since_high,
            'periods_since_low': periods_since_low
        }
    
    def compute_aroon_values(self, hc_lc_data):
        """
        Au, Ad: Compute Aroon Up and Aroon Down values
        """
        aroon_up = ((self.lookback_period - hc_lc_data['periods_since_high']) / self.lookback_period) * 100
        aroon_down = ((self.lookback_period - hc_lc_data['periods_since_low']) / self.lookback_period) * 100
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down
        }
    
    def generate_signal(self, aroon_oscillator):
        """
        Generate trading signal based on Aroon oscillator
        """
        if aroon_oscillator > 20:
            return 1  # Buy
        elif aroon_oscillator < -20:
            return -1  # Sell
        else:
            return 0  # Hold

def generate_realistic_price_stream(num_prices=50000, initial_price=100):
    """
    Generate realistic price stream with true randomness - each call is unique
    """
    # Use time-based seed for true randomness between calls
    import time
    random.seed(int(time.time() * 1000000) % 2**32)
    np.random.seed(int(time.time() * 1000000) % 2**32)
    
    current_price = initial_price
    base_drift = 0.0001
    base_volatility = 0.015
    
    # Random regime parameters (different each time)
    trend_duration = random.randint(800, 2500)
    volatility_duration = random.randint(300, 1200)
    current_trend = random.choice([-1, 0, 1]) * random.uniform(0.0005, 0.0015)
    current_vol_multiplier = random.uniform(0.7, 1.8)
    
    trend_counter = 0
    vol_counter = 0
    price_history = deque(maxlen=10)
    
    for i in range(num_prices):
        # Regime changes
        if trend_counter >= trend_duration:
            current_trend = random.choice([-1, 0, 1]) * random.uniform(0.0003, 0.0020)
            trend_duration = random.randint(800, 2500)
            trend_counter = 0
        
        if vol_counter >= volatility_duration:
            current_vol_multiplier = random.uniform(0.4, 2.5)
            volatility_duration = random.randint(300, 1200)
            vol_counter = 0
        
        # Price evolution
        drift = base_drift + current_trend
        volatility = base_volatility * current_vol_multiplier
        
        # Momentum using only recent history
        if len(price_history) >= 3:
            recent_momentum = (price_history[-1] - price_history[-3]) / price_history[-3]
            drift += recent_momentum * 0.05
        
        # Price update with true randomness
        random_shock = np.random.normal(0, 1)
        microstructure_noise = random.uniform(-0.0001, 0.0001)
        price_change = current_price * (drift + volatility * random_shock + microstructure_noise)
        
        current_price = max(current_price + price_change, 0.01)
        price_history.append(current_price)
        
        trend_counter += 1
        vol_counter += 1
        
        # Yield current price (true streaming)
        yield current_price

def compute_aroon_streaming_pipeline(price_generator, lookback_period=14):
    """
    Compute Aroon using streaming pipeline architecture - NO FUTURE DATA ACCESS
    """
    print("Computing Aroon with streaming pipeline...")
    
    processor = AroonPipelineProcessor(lookback_period)
    aroon_values = []
    signals = []
    
    # Process prices as they "arrive" one by one (streaming simulation)
    for current_price in price_generator:
        processor.price_window.append(current_price)
        
        if len(processor.price_window) < lookback_period:
            aroon_values.append(0)
            signals.append(0)
            continue
        
        # Process only the current window - no access to future prices
        window_copy = list(processor.price_window)
        
        # Pipeline stages using only current window
        hp_lp_result = processor.compute_high_low_prices(deque(window_copy))
        gt_lw_result = processor.compute_price_comparisons(deque(window_copy), hp_lp_result)
        hc_lc_result = processor.compute_high_low_counts(gt_lw_result)
        au_ad_result = processor.compute_aroon_values(hc_lc_result)
        
        aroon_oscillator = au_ad_result['aroon_up'] - au_ad_result['aroon_down']
        signal = processor.generate_signal(aroon_oscillator)
        
        aroon_values.append(aroon_oscillator)
        signals.append(signal)
    
    print("Aroon computation completed.")
    return aroon_values, signals

def plot_trading_simulation(prices, aroon_values, signals, start_idx=1000, end_idx=6000):
    """
    Create comprehensive visualization of trading simulation
    """
    print("Creating visualization...")
    
    # Extract plotting data
    plot_prices = prices[start_idx:end_idx]
    plot_aroon = aroon_values[start_idx:end_idx]
    plot_signals = signals[start_idx:end_idx]
    x_axis = range(len(plot_prices))
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Price chart
    ax1.plot(x_axis, plot_prices, 'b-', linewidth=1.2, label='Stock Price')
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Parallel Pipeline Aroon Trading Simulation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Aroon Oscillator
    ax2.plot(x_axis, plot_aroon, 'purple', linewidth=1.5, label='Aroon Oscillator')
    ax2.axhline(y=20, color='g', linestyle='--', alpha=0.8, linewidth=1.5, label='Buy Threshold (+20)')
    ax2.axhline(y=-20, color='r', linestyle='--', alpha=0.8, linewidth=1.5, label='Sell Threshold (-20)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4)
    ax2.fill_between(x_axis, plot_aroon, 0, alpha=0.2, color='purple')
    ax2.set_ylabel('Aroon Oscillator', fontsize=12, fontweight='bold')
    ax2.set_ylim(-100, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Trading signals
    buy_indices = [i for i, sig in enumerate(plot_signals) if sig == 1]
    sell_indices = [i for i, sig in enumerate(plot_signals) if sig == -1]
    hold_indices = [i for i, sig in enumerate(plot_signals) if sig == 0]
    
    if buy_indices:
        ax3.scatter(buy_indices, [1]*len(buy_indices), color='green', marker='^', 
                   s=30, alpha=0.8, label=f'Buy ({len(buy_indices)})', edgecolors='darkgreen')
    if sell_indices:
        ax3.scatter(sell_indices, [-1]*len(sell_indices), color='red', marker='v', 
                   s=30, alpha=0.8, label=f'Sell ({len(sell_indices)})', edgecolors='darkred')
    
    ax3.set_ylabel('Trading Signal', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['SELL', 'HOLD', 'BUY'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Trading statistics
    total_buy = sum(1 for s in plot_signals if s == 1)
    total_sell = sum(1 for s in plot_signals if s == -1)
    
    print("Graph plotted successfully.")
    print(f"Buy signals: {total_buy}, Sell signals: {total_sell}")

def process_streaming_simulation(num_prices=50000, lookback_period=14):
    """
    Process truly streaming simulation - store data as it arrives
    """
    print("=== TRUE Streaming Aroon Trading Simulation ===")
    
    processor = AroonPipelineProcessor(lookback_period)
    
    # Storage for streaming data (simulates real-time data storage)
    historical_prices = []
    aroon_values = []
    signals = []
    
    print("Processing streaming prices...")
    
    # Create single price stream - process each price ONCE as it "arrives"
    price_stream = generate_realistic_price_stream(num_prices)
    
    for i, current_price in enumerate(price_stream):
        # Store price as it arrives (like real trading system would)
        historical_prices.append(current_price)
        
        # Update sliding window
        processor.price_window.append(current_price)
        
        if len(processor.price_window) < lookback_period:
            aroon_values.append(0)
            signals.append(0)
            continue
        
        # Process using only current window (no future data)
        window_copy = list(processor.price_window)
        
        # Pipeline stages
        hp_lp_result = processor.compute_high_low_prices(deque(window_copy))
        gt_lw_result = processor.compute_price_comparisons(deque(window_copy), hp_lp_result)
        hc_lc_result = processor.compute_high_low_counts(gt_lw_result)
        au_ad_result = processor.compute_aroon_values(hc_lc_result)
        
        aroon_oscillator = au_ad_result['aroon_up'] - au_ad_result['aroon_down']
        signal = processor.generate_signal(aroon_oscillator)
        
        # Store results as they're computed
        aroon_values.append(aroon_oscillator)
        signals.append(signal)
        
        # Progress indicator
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} streaming prices...")
    
    print("Streaming simulation completed.")
    return historical_prices, aroon_values, signals

def main():
    """
    Execute complete TRUE streaming simulation
    """
    # Process streaming data once - store results
    prices, aroon_values, signals = process_streaming_simulation(50000, 14)
    
    start=time.time()
    # Plot using stored historical data
    plot_trading_simulation(prices, aroon_values, signals)
    end=time.time()
    print(f"Plotting completed in {end-start:.2f} seconds.")
    print("=== True streaming simulation completed ===")

if __name__ == "__main__":
    main()