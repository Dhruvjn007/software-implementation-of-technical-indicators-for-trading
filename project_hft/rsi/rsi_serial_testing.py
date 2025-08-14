# testing.py - separate file for testing
import numpy as np
import time
import random

def generate_random_prices(n):
    """Generate random price data in testing.py"""
    prices = np.zeros(n, dtype=np.float32)
    prices[0] = 100.0
    
    for i in range(1, n):
        change = (random.randint(0, 2000) - 1000) / 1000.0  # [-1.0, 1.0]
        prices[i] = prices[i-1] + change
    
    return prices

def test_rsi_performance():
    """Test RSI calculation performance"""
    N = 1000000
    PERIOD = 14
    
    # Generate prices in testing.py
    print("Generating random prices...")
    prices = generate_random_prices(N)
    
    # Test Cython version
    import rsi_serial_cython
    start_time = time.time()
    rsi_cython_result = rsi_serial_cython.calculate_rsi(prices, PERIOD)
    cython_time = time.time() - start_time
    
    print(f"Cython Time: {cython_time:.6f} sec")
    print(f"Sample RSI: {rsi_cython_result[N//2]:.2f} {rsi_cython_result[N//2 + 1]:.2f} {rsi_cython_result[N//2 + 2]:.2f}")
    
    # Test Python version for comparison
    start_time = time.time()
    rsi_python_result = calc_rsi_python(prices, PERIOD)
    python_time = time.time() - start_time
    
    print(f"Python Time: {python_time:.6f} sec")
    print(f"Speedup: {python_time/cython_time:.2f}x")

def calc_rsi_python(prices, period=14):
    """Pure Python RSI calculation for comparison"""
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float32)
    gain = loss = 0.0
    
    for i in range(1, period + 1):
        change = prices[i] - prices[i-1]
        if change > 0:
            gain += change
        else:
            loss -= change
    
    avg_gain = gain / period
    avg_loss = loss / period
    
    for i in range(period + 1, n):
        change = prices[i] - prices[i-1]
        g = change if change > 0 else 0
        l = -change if change < 0 else 0
        
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        
        if avg_loss == 0:
            rs = 100.0
        else:
            rs = avg_gain / avg_loss
        
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

if __name__ == "__main__":
    test_rsi_performance()