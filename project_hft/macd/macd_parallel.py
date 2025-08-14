import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from collections import deque

class ParallelMACDCalculator:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, chunk_size=1000):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.chunk_size = chunk_size
        
        # Pipeline stages
        self.price_buffer = deque(maxlen=chunk_size * 3)
        self.ema_buffer = deque(maxlen=chunk_size * 2)
        self.macd_buffer = deque(maxlen=chunk_size)
        
        # State tracking
        self.fast_ema_state = None
        self.slow_ema_state = None
        self.signal_ema_state = None
        self.processed_count = 0
        
        # Smoothing factors
        self.fast_alpha = 2 / (fast_period + 1)
        self.slow_alpha = 2 / (slow_period + 1)
        self.signal_alpha = 2 / (signal_period + 1)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = Lock()
        
        # Results storage
        self.results = {
            'macd_line': [],
            'signal_line': [],
            'histogram': [],
            'processing_times': []
        }

    def compute_ema_chunk(self, prices: List[float], alpha: float, 
                         initial_ema: Optional[float]) -> Tuple[List[float], float]:
        """Compute EMA for a chunk of prices"""
        emas = []
        current_ema = initial_ema
        
        for price in prices:
            if current_ema is None:
                current_ema = price
            else:
                current_ema = (price * alpha) + (current_ema * (1 - alpha))
            emas.append(current_ema)
        
        return emas, current_ema

    def compute_macd_chunk(self, fast_emas: List[float], slow_emas: List[float]) -> List[float]:
        """Compute MACD line for a chunk"""
        return [fast - slow for fast, slow in zip(fast_emas, slow_emas)]

    def compute_signal_chunk(self, macd_values: List[float], alpha: float, 
                           initial_signal: Optional[float]) -> Tuple[List[float], float]:
        """Compute signal line for a chunk of MACD values"""
        signals = []
        current_signal = initial_signal
        
        for macd in macd_values:
            if current_signal is None:
                current_signal = macd
            else:
                current_signal = (macd * alpha) + (current_signal * (1 - alpha))
            signals.append(current_signal)
        
        return signals, current_signal

    def process_chunk_parallel(self, price_chunk: List[float]) -> Dict:
        """Process a chunk of prices through the parallel pipeline"""
        start_time = time.time()
        
        # Stage 1: Compute EMAs in parallel
        fast_future = self.executor.submit(
            self.compute_ema_chunk, price_chunk, self.fast_alpha, self.fast_ema_state
        )
        slow_future = self.executor.submit(
            self.compute_ema_chunk, price_chunk, self.slow_alpha, self.slow_ema_state
        )
        
        # Wait for EMA computations
        fast_emas, new_fast_state = fast_future.result()
        slow_emas, new_slow_state = slow_future.result()
        
        # Update states
        self.fast_ema_state = new_fast_state
        self.slow_ema_state = new_slow_state
        
        # Stage 2: Compute MACD
        macd_future = self.executor.submit(self.compute_macd_chunk, fast_emas, slow_emas)
        macd_values = macd_future.result()
        
        # Stage 3: Compute Signal line
        signal_future = self.executor.submit(
            self.compute_signal_chunk, macd_values, self.signal_alpha, self.signal_ema_state
        )
        signal_values, new_signal_state = signal_future.result()
        self.signal_ema_state = new_signal_state
        
        # Stage 4: Compute Histogram
        histogram_future = self.executor.submit(
            lambda m, s: [macd - sig for macd, sig in zip(m, s)], 
            macd_values, signal_values
        )
        histogram_values = histogram_future.result()
        
        processing_time = time.time() - start_time
        
        return {
            'macd': macd_values,
            'signal': signal_values,
            'histogram': histogram_values,
            'processing_time': processing_time
        }

    def update_batch(self, price_batch: List[float]) -> Dict:
        """Update MACD calculation with a batch of prices while maintaining order"""
        with self.lock:
            # Add prices to buffer
            self.price_buffer.extend(price_batch)
            
            results_batch = {
                'macd': [],
                'signal': [],
                'histogram': [],
                'valid_from': self.processed_count
            }
            
            # Process in chunks while maintaining sequential order
            while len(self.price_buffer) >= self.chunk_size:
                # Extract chunk maintaining order
                chunk = [self.price_buffer.popleft() for _ in range(self.chunk_size)]
                
                # Process chunk
                chunk_results = self.process_chunk_parallel(chunk)
                
                # Only include results after warm-up period
                start_idx = max(0, self.slow_period - self.processed_count)
                if self.processed_count >= self.slow_period:
                    valid_results = {
                        'macd': chunk_results['macd'][start_idx:],
                        'signal': chunk_results['signal'][start_idx:],
                        'histogram': chunk_results['histogram'][start_idx:]
                    }
                    
                    results_batch['macd'].extend(valid_results['macd'])
                    results_batch['signal'].extend(valid_results['signal'])
                    results_batch['histogram'].extend(valid_results['histogram'])
                
                self.processed_count += self.chunk_size
                self.results['processing_times'].append(chunk_results['processing_time'])
            
            return results_batch

class ParallelTradingSignals:
    def __init__(self):
        self.previous_macd = None
        self.previous_signal = None
        self.position = 0
        self.signals = []
        self.lock = Lock()

    def generate_signals_batch(self, macd_batch: List[float], 
                              signal_batch: List[float]) -> List[int]:
        """Generate trading signals for a batch while maintaining sequence"""
        with self.lock:
            batch_signals = []
            
            for macd, signal in zip(macd_batch, signal_batch):
                signal_action = 0
                
                if self.previous_macd is not None and self.previous_signal is not None:
                    # Bullish crossover
                    if (self.previous_macd <= self.previous_signal and 
                        macd > signal and self.position <= 0):
                        signal_action = 1
                        self.position = 1
                    
                    # Bearish crossover
                    elif (self.previous_macd >= self.previous_signal and 
                          macd < signal and self.position >= 0):
                        signal_action = -1
                        self.position = -1
                
                batch_signals.append(signal_action)
                self.previous_macd = macd
                self.previous_signal = signal
            
            self.signals.extend(batch_signals)
            return batch_signals

def generate_streaming_prices(n_points: int, batch_size: int = 1000, 
                            initial_price: float = 100.0):
    """Generate streaming price data in batches"""
    np.random.seed(42)
    
    total_batches = (n_points + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_points)
        batch_length = end_idx - start_idx
        
        # Generate batch with some continuity
        if batch_idx == 0:
            base_price = initial_price
        else:
            # Continue from previous batch end
            base_price = initial_price * np.exp(np.sum(np.random.normal(0.0005, 0.02, start_idx)))
        
        # Generate returns for this batch
        returns = np.random.normal(0.0005, 0.02, batch_length)
        
        # Add trending periods randomly
        if np.random.random() < 0.1:  # 10% chance of trend
            trend_strength = np.random.normal(0, 0.002)
            returns += trend_strength
        
        # Convert to prices
        log_returns = np.cumsum(returns)
        prices = base_price * np.exp(log_returns - log_returns[0])
        
        yield prices.tolist()

def compute_parallel_macd_signals(price_stream, chunk_size: int = 1000):
    """Compute MACD and trading signals using parallel processing"""
    macd_calc = ParallelMACDCalculator(chunk_size=chunk_size)
    signal_gen = ParallelTradingSignals()
    
    all_results = {
        'prices': [],
        'macd_line': [],
        'signal_line': [],
        'histogram': [],
        'trading_signals': [],
        'processing_times': []
    }
    
    batch_count = 0
    total_processing_time = 0
    
    for price_batch in price_stream:
        batch_count += 1
        
        # Store prices
        all_results['prices'].extend(price_batch)
        
        # Compute MACD indicators
        macd_results = macd_calc.update_batch(price_batch)
        
        # Generate trading signals if we have MACD data
        if macd_results['macd']:
            trading_signals = signal_gen.generate_signals_batch(
                macd_results['macd'], macd_results['signal']
            )
            
            all_results['macd_line'].extend(macd_results['macd'])
            all_results['signal_line'].extend(macd_results['signal'])
            all_results['histogram'].extend(macd_results['histogram'])
            all_results['trading_signals'].extend(trading_signals)
        
        # Track processing times
        if macd_calc.results['processing_times']:
            total_processing_time += macd_calc.results['processing_times'][-1]
    
    # Clean up executor
    macd_calc.executor.shutdown(wait=True)
    
    # Add performance metrics
    all_results['total_batches'] = batch_count
    all_results['avg_processing_time'] = total_processing_time / max(1, len(macd_calc.results['processing_times']))
    
    return all_results

def plot_parallel_results(results):
    """Plot results from parallel MACD computation"""
    prices = np.array(results['prices'])
    macd_line = np.array(results['macd_line']) if results['macd_line'] else np.array([])
    signal_line = np.array(results['signal_line']) if results['signal_line'] else np.array([])
    histogram = np.array(results['histogram']) if results['histogram'] else np.array([])
    trading_signals = np.array(results['trading_signals']) if results['trading_signals'] else np.array([])
    
    # Plot last 5000 points for visibility
    start_idx = max(0, len(prices) - 5000)
    plot_indices = np.arange(len(prices))[start_idx:]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Price and trading signals
    ax1.plot(plot_indices, prices[start_idx:], 'b-', linewidth=1, label='Price')
    
    if len(trading_signals) > 0:
        # Align signals with price indices
        signal_offset = len(prices) - len(trading_signals)
        signal_indices = plot_indices - signal_offset
        valid_signal_mask = (signal_indices >= 0) & (signal_indices < len(trading_signals))
        
        if np.any(valid_signal_mask):
            valid_plot_indices = plot_indices[valid_signal_mask]
            valid_signal_indices = signal_indices[valid_signal_mask].astype(int)
            
            buy_mask = trading_signals[valid_signal_indices] == 1
            sell_mask = trading_signals[valid_signal_indices] == -1
            
            if np.any(buy_mask):
                ax1.scatter(valid_plot_indices[buy_mask], prices[valid_plot_indices[buy_mask]], 
                           color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            if np.any(sell_mask):
                ax1.scatter(valid_plot_indices[sell_mask], prices[valid_plot_indices[sell_mask]], 
                           color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.set_title('Price Chart with Trading Signals (Parallel Processing)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MACD and Signal lines
    if len(macd_line) > 0:
        macd_offset = len(prices) - len(macd_line)
        macd_plot_indices = plot_indices - macd_offset
        valid_macd_mask = (macd_plot_indices >= 0) & (macd_plot_indices < len(macd_line))
        
        if np.any(valid_macd_mask):
            valid_macd_plot_indices = plot_indices[valid_macd_mask]
            valid_macd_indices = macd_plot_indices[valid_macd_mask].astype(int)
            
            ax2.plot(valid_macd_plot_indices, macd_line[valid_macd_indices], 
                    'b-', linewidth=1, label='MACD Line')
            ax2.plot(valid_macd_plot_indices, signal_line[valid_macd_indices], 
                    'r-', linewidth=1, label='Signal Line')
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.set_title('MACD and Signal Line')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram
    if len(histogram) > 0:
        hist_offset = len(prices) - len(histogram)
        hist_plot_indices = plot_indices - hist_offset
        valid_hist_mask = (hist_plot_indices >= 0) & (hist_plot_indices < len(histogram))
        
        if np.any(valid_hist_mask):
            valid_hist_plot_indices = plot_indices[valid_hist_mask]
            valid_hist_indices = hist_plot_indices[valid_hist_mask].astype(int)
            valid_hist_values = histogram[valid_hist_indices]
            
            colors = ['red' if x < 0 else 'green' for x in valid_hist_values]
            ax3.bar(valid_hist_plot_indices, valid_hist_values, color=colors, 
                   alpha=0.7, width=1)
    
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Histogram')
    ax3.set_xlabel('Time')
    ax3.set_title('MACD Histogram')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total_signals = np.sum(np.abs(trading_signals)) if len(trading_signals) > 0 else 0
    buy_signals = np.sum(trading_signals == 1) if len(trading_signals) > 0 else 0
    sell_signals = np.sum(trading_signals == -1) if len(trading_signals) > 0 else 0
    
    print(f"Parallel Processing Results:")
    print(f"Total price points: {len(prices):,}")
    print(f"Total batches processed: {results.get('total_batches', 0)}")
    print(f"Total trading signals: {total_signals}")
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")

if __name__ == "__main__":
    # Run parallel MACD simulation
    price_stream = generate_streaming_prices(1_000_000, batch_size=1000)
    start=time.time()
    results = compute_parallel_macd_signals(price_stream, chunk_size=1000)
    end=time.time()
    print(f"Total processing time: {end-start:.2f} seconds")
    plot_parallel_results(results)