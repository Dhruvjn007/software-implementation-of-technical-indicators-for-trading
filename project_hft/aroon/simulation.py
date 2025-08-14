import numpy as np
import matplotlib.pyplot as plt
import aroon_simulator  # Import compiled Cython module

def plot_high_performance_results(prices, aroon_values, signals, start_idx=1000, end_idx=6000):
    print("Creating high-performance visualization")
    
    # Extract plotting data
    plot_prices = prices[start_idx:end_idx]
    plot_aroon = aroon_values[start_idx:end_idx]
    plot_signals = signals[start_idx:end_idx]
    x_axis = np.arange(len(plot_prices))
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Price chart
    ax1.plot(x_axis, plot_prices, 'b-', linewidth=1, alpha=0.8, label='High-Freq Price')
    ax1.set_ylabel('Price ', fontsize=12, fontweight='bold')
    ax1.set_title('High-Performance Cython Aroon Trading Simulation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Aroon Oscillator
    ax2.plot(x_axis, plot_aroon, 'purple', linewidth=1.2, alpha=0.9, label='Aroon Oscillator')
    ax2.axhline(y=20, color='g', linestyle='--', alpha=0.8, linewidth=1.5, label='Buy Threshold')
    ax2.axhline(y=-20, color='r', linestyle='--', alpha=0.8, linewidth=1.5, label='Sell Threshold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4)
    ax2.fill_between(x_axis, plot_aroon, 0, alpha=0.15, color='purple')
    ax2.set_ylabel('Aroon Oscillator', fontsize=12, fontweight='bold')
    ax2.set_ylim(-100, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Trading signals with proper conditional plotting
    buy_indices = np.where(plot_signals == 1)[0]
    sell_indices = np.where(plot_signals == -1)[0]
    
    if len(buy_indices) > 0:
        ax3.scatter(buy_indices, np.ones(len(buy_indices)), color='green', marker='^', 
                   s=25, alpha=0.7, label=f'Buy ({len(buy_indices)})', edgecolors='darkgreen')
    if len(sell_indices) > 0:
        ax3.scatter(sell_indices, -np.ones(len(sell_indices)), color='red', marker='v', 
                   s=25, alpha=0.7, label=f'Sell ({len(sell_indices)})', edgecolors='darkred')
    
    # Always show hold line for reference
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Hold Line')
    
    ax3.set_ylabel('Signal', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['SELL', 'HOLD', 'BUY'])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Performance statistics
    total_buy = np.sum(plot_signals == 1)
    total_sell = np.sum(plot_signals == -1)
    
    print("Visualization completed.")
    print(f"Performance: Buy={total_buy}, Sell={total_sell}")

def main():
    import time
    
    start_time = time.time()
    
    # Run simulation
    prices, aroon_values, signals = aroon_simulator.run_high_performance_simulation(10000000)
    
    execution_time = time.time() - start_time
    
    # Plot results
    plot_high_performance_results(prices, aroon_values, signals)
    
    print(f"Execution time: {execution_time:.3f} seconds")
    print("=== High-performance simulation completed ===")

if __name__ == "__main__":
    main()