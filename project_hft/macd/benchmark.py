import numpy as np
import time
import matplotlib.pyplot as plt
from macd_fast import FastMACDProcessor

class PythonMACDReference:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period  
        self.signal_period = signal_period
        self.fast_alpha = 2 / (fast_period + 1)
        self.slow_alpha = 2 / (slow_period + 1)
        self.signal_alpha = 2 / (signal_period + 1)
        self.reset()
    
    def reset(self):
        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self.count = 0
    
    def update(self, price):
        self.count += 1
        
        if self.fast_ema is None:
            self.fast_ema = price
        else:
            self.fast_ema = self.fast_alpha * price + (1 - self.fast_alpha) * self.fast_ema
            
        if self.slow_ema is None:
            self.slow_ema = price
        else:
            self.slow_ema = self.slow_alpha * price + (1 - self.slow_alpha) * self.slow_ema
        
        if self.count >= self.slow_period:
            macd = self.fast_ema - self.slow_ema
            
            if self.signal_ema is None:
                self.signal_ema = macd
            else:
                self.signal_ema = self.signal_alpha * macd + (1 - self.signal_alpha) * self.signal_ema
            
            if self.count >= (self.slow_period + self.signal_period - 1):
                histogram = macd - self.signal_ema
                return macd, self.signal_ema, histogram, True
        
        return 0.0, 0.0, 0.0, False

def generate_test_prices(n_points: int) -> np.ndarray:
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_points)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices.astype(np.float64)

def plot_latency_distribution():
    prices = generate_test_prices(10_000)
    cython_proc = FastMACDProcessor()
    
    latencies = []
    for price in prices:
        start = time.perf_counter()
        cython_proc.update(price)
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)
    
    latencies = np.array(latencies)
    print(f"Average latency: {np.mean(latencies):.2f}μs")
    
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Latency (microseconds)')
    plt.ylabel('Frequency')
    plt.title('MACD Update Latency Distribution')
    plt.axvline(np.median(latencies), color='red', linestyle='--', 
                label=f'Median: {np.median(latencies):.1f}μs')
    plt.axvline(np.percentile(latencies, 95), color='orange', linestyle='--',
                label=f'95th percentile: {np.percentile(latencies, 95):.1f}μs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_latency_distribution()