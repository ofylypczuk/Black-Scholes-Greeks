
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from bsm_pricer import OptionPricer, OptionVisualizer

def run_verification():
    print("Running verification...")
    
    # 1. Text Output Verification
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    pricer = OptionPricer(S, K, T, r, sigma)
    prices = pricer.calculate_price()
    greeks = pricer.calculate_greeks()
    
    print(f"Call: {prices['call']:.4f}")
    print(f"Put: {prices['put']:.4f}")
    print(f"Delta Call: {greeks['call']['delta']:.4f}")
    
    # 2. Plot Verification (Save instead of Show)
    # Monkey patch plt.show to save
    original_show = plt.show
    
    def save_plot_1():
        plt.savefig('heatmap_gamma.png')
        plt.close()
        print("Saved heatmap_gamma.png")

    def save_plot_2():
        plt.savefig('surface_delta.png')
        plt.close()
        print("Saved surface_delta.png")

    # Plot 1
    plt.show = save_plot_1
    OptionVisualizer.plot_heatmap(pricer, Greek='Gamma', OptionType='call')
    
    # Plot 2
    plt.show = save_plot_2
    OptionVisualizer.plot_greeks_surface(pricer, Greek='Delta', OptionType='call')

if __name__ == "__main__":
    run_verification()
