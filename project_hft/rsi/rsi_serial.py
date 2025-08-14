import time
import random
import matplotlib.pyplot as plt
import numpy as np

def generate_realistic_prices(n_points=1000, initial_price=100, trend=0.0001, volatility=0.02):
    """
    Generate realistic price data using random walk with trend and volatility
    
    Args:
        n_points: Number of price points to generate
        initial_price: Starting price
        trend: Daily trend (positive for upward trend)
        volatility: Price volatility (standard deviation of returns)
    """
    prices = [initial_price]
    
    for i in range(1, n_points):
        # Random walk with trend and volatility
        random_change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + random_change)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, 0.01)
        prices.append(new_price)
    
    return prices

def rsi_serial(prices, period=14):
    deltas = []
    gains = []
    losses = []
    avg_gain = []
    avg_loss = []
    rsi_values = []
    
    # Stage 1: Calculate price differences
    for i in range(1, len(prices)):
        deltas.append(prices[i] - prices[i - 1])
    
    # Stage 2: Separate gains and losses
    for d in deltas:
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    
    # Stage 3: Calculate initial average gain and loss
    gain_sum = sum(gains[:period])
    loss_sum = sum(losses[:period])
    ema_gain = gain_sum / period
    ema_loss = loss_sum / period
    avg_gain.append(ema_gain)
    avg_loss.append(ema_loss)
    
    # Stage 4: Calculate first RSI value
    rs = ema_gain / ema_loss if ema_loss != 0 else float('inf')
    rsi_values = [None] * period  # padding for warm-up period
    rsi_values.append(100 - (100 / (1 + rs)))
    
    # Continue EMA calculation and RSI computation for remaining periods
    for i in range(period, len(gains)):  # Fixed: should be len(gains), not len(prices)-1
        ema_gain = (ema_gain * (period - 1) + gains[i]) / period
        ema_loss = (ema_loss * (period - 1) + losses[i]) / period
        avg_gain.append(ema_gain)
        avg_loss.append(ema_loss)
        
        rs = ema_gain / ema_loss if ema_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return rsi_values

def generate_trading_signals(rsi_values, oversold_threshold=30, overbought_threshold=70):
    """
    Generate buy/sell/hold signals based on RSI values
    
    Args:
        rsi_values: List of RSI values
        oversold_threshold: RSI threshold for buy signal (default 30)
        overbought_threshold: RSI threshold for sell signal (default 70)
    
    Returns:
        List of signals: 'buy', 'sell', or 'hold'
    """
    signals = []
    
    for rsi in rsi_values:
        if rsi is None:
            signals.append('hold')
        elif rsi <= oversold_threshold:
            signals.append('buy')
        elif rsi >= overbought_threshold:
            signals.append('sell')
        else:
            signals.append('hold')
    
    return signals

def plot_rsi_analysis(prices, rsi_values, signals, period=14):
    """
    Plot prices, RSI, and trading signals
    
    Args:
        prices: List of price values
        rsi_values: List of RSI values
        signals: List of trading signals
        period: RSI period for title
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Price chart with buy/sell signals
    ax1.plot(prices, label='Price', color='blue', linewidth=1)
    
    # Add buy and sell signals to price chart
    buy_points = []
    sell_points = []
    buy_prices = []
    sell_prices = []
    
    for i, signal in enumerate(signals):
        if signal == 'buy':
            buy_points.append(i)
            buy_prices.append(prices[i])
        elif signal == 'sell':
            sell_points.append(i)
            sell_prices.append(prices[i])
    
    ax1.scatter(buy_points, buy_prices, color='green', marker='^', s=50, label='Buy Signal', alpha=0.7)
    ax1.scatter(sell_points, sell_prices, color='red', marker='v', s=50, label='Sell Signal', alpha=0.7)
    
    ax1.set_ylabel('Price')
    ax1.set_title(f'Price Chart with RSI Trading Signals (Period={period})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI chart with overbought/oversold levels
    # Filter out None values for plotting
    rsi_plot_values = []
    rsi_indices = []
    for i, rsi in enumerate(rsi_values):
        if rsi is not None:
            rsi_plot_values.append(rsi)
            rsi_indices.append(i)
    
    ax2.plot(rsi_indices, rsi_plot_values, label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Neutral (50)')
    
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Time Period')
    ax2.set_title(f'RSI Indicator (Period={period})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Generating realistic price data...")
    
    # Generate realistic prices with slight upward trend
    realistic_prices = generate_realistic_prices(
        n_points=1000, 
        initial_price=100, 
        trend=0.0005,  # Slight upward trend
        volatility=0.015  # 1.5% daily volatility
    )
    
    print("Computing RSI...")
    start_time = time.time()
    
    # Calculate RSI
    rsi_output = rsi_serial(realistic_prices, period=14)
    
    end_time = time.time()
    print(f"RSI computation completed in {end_time - start_time:.6f} seconds")
    
    print("Generating trading signals...")
    # Generate trading signals
    trading_signals = generate_trading_signals(rsi_output)
    
    # Count signals
    signal_counts = {
        'buy': trading_signals.count('buy'),
        'sell': trading_signals.count('sell'),
        'hold': trading_signals.count('hold')
    }
    
    print(f"Trading signals generated:")
    print(f"  Buy signals: {signal_counts['buy']}")
    print(f"  Sell signals: {signal_counts['sell']}")
    print(f"  Hold signals: {signal_counts['hold']}")
    
    print("Plotting results...")
    # Plot the results
    plot_rsi_analysis(realistic_prices, rsi_output, trading_signals)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()