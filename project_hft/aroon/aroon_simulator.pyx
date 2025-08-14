import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log, exp
from libc.stdlib cimport rand, srand, RAND_MAX
import time

# Type definitions
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class HighPerformancePriceGenerator:
    """
    Ultra-fast price generator using Cython - true streaming with no future access
    """
    cdef:
        double current_price
        double base_drift
        double base_volatility
        double current_trend
        double current_vol_multiplier
        int trend_duration, volatility_duration
        int trend_counter, vol_counter
        double[10] price_history  # Fixed-size circular buffer
        int history_index, history_size
        int total_prices_generated
        
    def __init__(self, double initial_price = 100.0):
        # Initialize with time-based random seed for true randomness
        srand(int(time.time() * 1000000) % 2147483647)
        
        self.current_price = initial_price
        self.base_drift = 0.0001
        self.base_volatility = 0.015
        
        # Random regime initialization
        self.trend_duration = 800 + (rand() % 1700)  # 800-2500
        self.volatility_duration = 300 + (rand() % 900)  # 300-1200
        self.current_trend = (rand() % 3 - 1) * (0.0005 + (rand() / <double>RAND_MAX) * 0.001)
        self.current_vol_multiplier = 0.7 + (rand() / <double>RAND_MAX) * 1.1
        
        self.trend_counter = 0
        self.vol_counter = 0
        self.history_index = 0
        self.history_size = 0
        self.total_prices_generated = 0
        
        # Initialize price history
        for i in range(10):
            self.price_history[i] = initial_price
    
    @cython.cdivision(True)
    cdef double generate_next_price(self):
        """
        Generate next price using only current state and history - NO FUTURE DATA
        """
        cdef double drift, volatility, momentum, random_shock, microstructure_noise
        cdef double price_change, old_price
        cdef int momentum_lookback, hist_idx
        
        # Regime changes based on counters
        if self.trend_counter >= self.trend_duration:
            self.current_trend = ((rand() % 3) - 1) * (0.0005 + (rand() / <double>RAND_MAX) * 0.0015)
            self.trend_duration = 800 + (rand() % 1700)
            self.trend_counter = 0
        
        if self.vol_counter >= self.volatility_duration:
            self.current_vol_multiplier = 0.5 + (rand() / <double>RAND_MAX) * 2.0
            self.volatility_duration = 300 + (rand() % 900)
            self.vol_counter = 0
        
        # Calculate momentum from recent history only
        momentum = 0.0
        if self.history_size >= 3:
            hist_idx = (self.history_index - 3 + 10) % 10
            old_price = self.price_history[hist_idx]
            if old_price > 0.01:
                momentum = (self.current_price - old_price) / old_price
                momentum = max(min(momentum, 0.1), -0.1)  # Cap momentum
        
        # Price evolution parameters
        drift = self.base_drift + self.current_trend + momentum * 0.03
        volatility = self.base_volatility * self.current_vol_multiplier
        
        # Random components - fix Box-Muller implementation
        random_shock = self._box_muller_normal()
        microstructure_noise = -0.0002 + (rand() / <double>RAND_MAX) * 0.0004
        
        # Price update with proper scaling
        price_change = self.current_price * (drift + volatility * random_shock) + microstructure_noise
        self.current_price = max(self.current_price + price_change, 1.0)
        
        # Update circular buffer history
        self.price_history[self.history_index] = self.current_price
        self.history_index = (self.history_index + 1) % 10
        if self.history_size < 10:
            self.history_size += 1
        
        self.trend_counter += 1
        self.vol_counter += 1
        self.total_prices_generated += 1
        
        return self.current_price
    
    @cython.cdivision(True)
    cdef double _box_muller_normal(self):
        """Fast Box-Muller normal distribution generator"""
        cdef double u1, u2, z
        u1 = (rand() + 1.0) / (<double>RAND_MAX + 2.0)  # Avoid log(0)
        u2 = (rand() + 1.0) / (<double>RAND_MAX + 2.0)  # Avoid log(0)
        z = sqrt(-2.0 * log(u1)) * np.cos(2.0 * np.pi * u2)
        return z

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FastAroonProcessor:
    """
    High-performance Aroon processor using circular buffers
    """
    cdef:
        double[14] price_window  # Fixed-size circular buffer
        int window_index, window_size
        int lookback_period
        
    def __init__(self, int lookback_period = 14):
        self.lookback_period = lookback_period
        self.window_index = 0
        self.window_size = 0
        
        # Initialize window
        for i in range(14):
            self.price_window[i] = 0.0
    
    @cython.cdivision(True)
    cdef tuple process_price(self, double price):
        """
        Process single price through Aroon pipeline - returns (aroon_osc, signal)
        """
        cdef double highest_high, lowest_low
        cdef int periods_since_high, periods_since_low
        cdef double aroon_up, aroon_down, aroon_oscillator
        cdef int signal, i, actual_index
        
        # Add price to circular buffer
        self.price_window[self.window_index] = price
        self.window_index = (self.window_index + 1) % self.lookback_period
        if self.window_size < self.lookback_period:
            self.window_size += 1
        
        # Not enough data yet
        if self.window_size < self.lookback_period:
            return (0.0, 0)
        
        # Find highest high and lowest low in current window
        highest_high = self.price_window[0]
        lowest_low = self.price_window[0]
        
        for i in range(self.lookback_period):
            if self.price_window[i] > highest_high:
                highest_high = self.price_window[i]
            if self.price_window[i] < lowest_low:
                lowest_low = self.price_window[i]
        
        # Find periods since highest high and lowest low (search backwards)
        periods_since_high = self.lookback_period - 1
        periods_since_low = self.lookback_period - 1
        
        for i in range(self.lookback_period):
            actual_index = (self.window_index - 1 - i + self.lookback_period) % self.lookback_period
            
            # Check for high (most recent occurrence wins)
            if self.price_window[actual_index] == highest_high and periods_since_high >= i:
                periods_since_high = i
                
            # Check for low (most recent occurrence wins)  
            if self.price_window[actual_index] == lowest_low and periods_since_low >= i:
                periods_since_low = i
        
        # Calculate Aroon values
        aroon_up = ((self.lookback_period - periods_since_high) / <double>self.lookback_period) * 100.0
        aroon_down = ((self.lookback_period - periods_since_low) / <double>self.lookback_period) * 100.0
        aroon_oscillator = aroon_up - aroon_down
        
        # Generate signal with more sensitive thresholds for testing
        if aroon_oscillator > 15.0:
            signal = 1  # Buy
        elif aroon_oscillator < -15.0:
            signal = -1  # Sell
        else:
            signal = 0  # Hold
        
        return (aroon_oscillator, signal)

def run_high_performance_simulation(int num_prices = 100000):
    """
    Execute high-performance streaming simulation
    """
    print("High-performance Cython simulation started...")
    
    # Initialize components
    cdef HighPerformancePriceGenerator price_gen = HighPerformancePriceGenerator(100.0)
    cdef FastAroonProcessor aroon_proc = FastAroonProcessor(14)
    
    # Pre-allocate arrays for results
    cdef np.ndarray[DTYPE_t, ndim=1] prices = np.zeros(num_prices, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] aroon_values = np.zeros(num_prices, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] signals = np.zeros(num_prices, dtype=np.int32)
    
    cdef double current_price, aroon_osc
    cdef int signal, i
    cdef tuple result
    
    # Process streaming prices - each price unknown until generated
    for i in range(num_prices):
        # Generate next price (streaming - no future knowledge)
        current_price = price_gen.generate_next_price()
        prices[i] = current_price
        
        # Process through Aroon pipeline immediately
        result = aroon_proc.process_price(current_price)
        aroon_osc = result[0]
        signal = result[1]
        
        aroon_values[i] = aroon_osc
        signals[i] = signal
        
        # Progress indicator (minimal)
        if i > 0 and i % 25000 == 0:
            print(f"Processed {i} prices...")
    
    print("High-performance simulation completed.")
    return np.array(prices), np.array(aroon_values), np.array(signals)
