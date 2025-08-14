# File 1: macd_fast.pyx
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf
from cython cimport floating

# Type definitions
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef struct MACDResult:
    double macd
    double signal
    double histogram
    bint is_valid

cdef class FastMACDProcessor:
    """Ultra-fast MACD processor for real-time trading"""
    
    # State variables
    cdef double fast_ema
    cdef double slow_ema  
    cdef double signal_ema
    
    # Configuration
    cdef double fast_alpha
    cdef double slow_alpha
    cdef double signal_alpha
    cdef int fast_period
    cdef int slow_period
    cdef int signal_period
    
    # Counters
    cdef long long count
    cdef int warmup_period
    cdef bint fast_initialized
    cdef bint slow_initialized
    cdef bint signal_initialized
    
    def __init__(self, int fast_period=12, int slow_period=26, int signal_period=9):
        """Initialize MACD processor with specified periods"""
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Calculate smoothing factors
        self.fast_alpha = 2.0 / (fast_period + 1)
        self.slow_alpha = 2.0 / (slow_period + 1)
        self.signal_alpha = 2.0 / (signal_period + 1)
        
        # Initialize state
        self.reset()
    
    cdef void reset(self) nogil:
        """Reset all state variables"""
        self.fast_ema = 0.0
        self.slow_ema = 0.0
        self.signal_ema = 0.0
        self.count = 0
        self.warmup_period = self.slow_period + self.signal_period
        self.fast_initialized = False
        self.slow_initialized = False
        self.signal_initialized = False
    
    cdef inline MACDResult update_single_nogil(self, double price) nogil:
        """Update MACD with single price point - no GIL version"""
        cdef MACDResult result
        cdef double macd_value
        
        # Initialize result
        result.macd = 0.0
        result.signal = 0.0
        result.histogram = 0.0
        result.is_valid = False
        
        # Validate input
        if isnan(price) or isinf(price):
            return result
        
        self.count += 1
        
        # Update fast EMA
        if not self.fast_initialized:
            self.fast_ema = price
            self.fast_initialized = True
        else:
            self.fast_ema = self.fast_alpha * price + (1.0 - self.fast_alpha) * self.fast_ema
        
        # Update slow EMA
        if not self.slow_initialized:
            self.slow_ema = price
            self.slow_initialized = True
        else:
            self.slow_ema = self.slow_alpha * price + (1.0 - self.slow_alpha) * self.slow_ema
        
        # Calculate MACD line
        if self.count >= self.slow_period:
            macd_value = self.fast_ema - self.slow_ema
            result.macd = macd_value
            
            # Update signal line
            if not self.signal_initialized:
                self.signal_ema = macd_value
                self.signal_initialized = True
            else:
                self.signal_ema = self.signal_alpha * macd_value + (1.0 - self.signal_alpha) * self.signal_ema
            
            result.signal = self.signal_ema
            
            # Calculate histogram
            if self.count >= self.warmup_period:
                result.histogram = macd_value - self.signal_ema
                result.is_valid = True
        
        return result
    
    cpdef tuple update(self, double price):
        """Update MACD with single price point - Python interface"""
        cdef MACDResult result = self.update_single_nogil(price)
        return (result.macd, result.signal, result.histogram, result.is_valid)
    


cdef class FastTradingSignals:
    """Ultra-fast trading signal generator"""
    
    cdef double prev_macd
    cdef double prev_signal
    cdef int position  # -1: short, 0: neutral, 1: long
    cdef bint initialized
    
    def __init__(self):
        self.reset()
    
    cdef void reset(self) nogil:
        """Reset signal state"""
        self.prev_macd = 0.0
        self.prev_signal = 0.0
        self.position = 0
        self.initialized = False
    
    cdef inline int generate_signal_nogil(self, double macd, double signal) nogil:
        """Generate trading signal - no GIL version"""
        cdef int trade_signal = 0
        
        if not self.initialized:
            self.prev_macd = macd
            self.prev_signal = signal
            self.initialized = True
            return 0
        
        # Bullish crossover: MACD crosses above Signal
        if self.prev_macd <= self.prev_signal and macd > signal and self.position <= 0:
            trade_signal = 1  # Buy
            self.position = 1
        
        # Bearish crossover: MACD crosses below Signal  
        elif self.prev_macd >= self.prev_signal and macd < signal and self.position >= 0:
            trade_signal = -1  # Sell
            self.position = -1
        
        # Update previous values
        self.prev_macd = macd
        self.prev_signal = signal
        
        return trade_signal
    
    cpdef int update(self, double macd, double signal):
        """Generate trading signal - Python interface"""
        return self.generate_signal_nogil(macd, signal)
    


# Convenience functions for Python users
def create_macd_processor(fast_period=12, slow_period=26, signal_period=9):
    """Create a new MACD processor instance"""
    return FastMACDProcessor(fast_period, slow_period, signal_period)

def create_signal_generator():
    """Create a new trading signal generator"""
    return FastTradingSignals()

# Batch processing function for maximum performance
cpdef tuple process_prices_streaming(double[:] prices, 
                                   int fast_period=12, 
                                   int slow_period=26, 
                                   int signal_period=9):
    """Process price array one-by-one to simulate streaming - DEMO ONLY"""
    cdef int n = prices.shape[0]
    cdef list macd_out = []
    cdef list signal_out = []
    cdef list histogram_out = []
    cdef list valid_out = []
    cdef list signals_out = []
    
    # Create processors
    cdef FastMACDProcessor macd_proc = FastMACDProcessor(fast_period, slow_period, signal_period)
    cdef FastTradingSignals signal_gen = FastTradingSignals()
    cdef int i
    cdef tuple macd_result
    cdef int trade_signal
    
    # Process each price sequentially (simulating real-time arrival)
    for i in range(n):
        # Process single price as it "arrives"
        macd_result = macd_proc.update(prices[i])
        macd_out.append(macd_result[0])
        signal_out.append(macd_result[1]) 
        histogram_out.append(macd_result[2])
        valid_out.append(macd_result[3])
        
        # Generate trading signal if valid
        if macd_result[3]:
            trade_signal = signal_gen.update(macd_result[0], macd_result[1])
        else:
            trade_signal = 0
        signals_out.append(trade_signal)
    
    return (np.array(macd_out), np.array(signal_out), np.array(histogram_out), 
            np.array(valid_out), np.array(signals_out))