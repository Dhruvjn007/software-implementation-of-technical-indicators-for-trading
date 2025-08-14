# rsi_cython.pyx
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
import time as py_time

cdef int PERIOD = 14
cdef int N = 1000000

cdef void calc_rsi_serial(cnp.float32_t[:] prices, cnp.float32_t[:] rsi, int n, int period):
    """Calculate RSI using Cython for speed"""
    cdef float gain = 0.0, loss = 0.0
    cdef int i,j
    cdef float change, avg_gain, avg_loss, g, l, rs
    j=period+1
    
    # Calculate initial gains and losses
    for i in range(1, j):
        change = prices[i] - prices[i-1]
        if change > 0:
            gain += change
        else:
            loss -= change
    
    avg_gain = gain / period
    avg_loss = loss / period
    
    # Calculate RSI for remaining periods
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

def calculate_rsi(cnp.float32_t[:] prices, int period=14):
    """Public function to calculate RSI - called from testing.py"""
    cdef int n = len(prices)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] rsi = np.zeros(n, dtype=np.float32)
    
    calc_rsi_serial(prices, rsi, n, period)
    return rsi