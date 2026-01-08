import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from typing import Dict, Tuple, Optional

# Set plotting style for professional aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

class OptionPricer:
    """
    A professional-grade class for pricing European Options using the Black-Scholes-Merton (BSM) model.
    This class supports both Call and Put options and calculates analytical Greeks.
    
    Attributes:
        S (float): Current Spot Price of the underlying asset.
        K (float): Strike Price of the option.
        T (float): Time to Maturity in years.
        r (float): Risk-free interest rate (annualized, continuous compounding).
        sigma (float): Volatility of the underlying asset (annualized standard deviation).
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    @property
    def _d1(self) -> float:
        """
        Calculates d1 term used in BSM formulas.
        
        .. math::
            d_1 = \\frac{\\ln(S/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}
        """
        if self.T == 0:
            return np.inf if self.S > self.K else -np.inf
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    @property
    def _d2(self) -> float:
        """
        Calculates d2 term used in BSM formulas.
        
        .. math::
            d_2 = d_1 - \\sigma\\sqrt{T}
        """
        return self._d1 - self.sigma * np.sqrt(self.T)

    def calculate_price(self) -> Dict[str, float]:
        """
        Calculates the fair theoretical price of Call and Put options.

        .. math::
            C = S N(d_1) - K e^{-rT} N(d_2)
            P = K e^{-rT} N(-d_2) - S N(-d_1)
        
        Returns:
            Dict[str, float]: Dictionary containing 'call' and 'put' prices.
        """
        d1 = self._d1
        d2 = self._d2
        
        call_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        return {'call': call_price, 'put': put_price}

    def calculate_greeks(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates the Greeks (partial derivatives) for Call and Put options.
        
        Formulas implemented:
        - **Delta** (Sensitivity to Spot Price):
            Call: :math:`N(d_1)`
            Put: :math:`N(d_1) - 1`
        - **Gamma** (Sensitivity to Delta/Spot Change):
            Both: :math:`\\frac{N'(d_1)}{S \\sigma \\sqrt{T}}`
        - **Vega** (Sensitivity to Volatility):
            Both: :math:`S \\sqrt{T} N'(d_1)`
        - **Theta** (Sensitivity to Time Decay):
            Call: :math:`-\\frac{S N'(d_1) \\sigma}{2\\sqrt{T}} - r K e^{-rT} N(d_2)`
            Put: :math:`-\\frac{S N'(d_1) \\sigma}{2\\sqrt{T}} + r K e^{-rT} N(-d_2)`
        - **Rho** (Sensitivity to Interest Rate):
            Call: :math:`K T e^{-rT} N(d_2)`
            Put: :math:`-K T e^{-rT} N(-d_2)`

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with Greeks for 'call' and 'put'.
        """
        d1 = self._d1
        d2 = self._d2
        
        # Helper terms
        N_prime_d1 = norm.pdf(d1)
        sqrt_T = np.sqrt(self.T)
        
        # Common Greeks
        gamma = N_prime_d1 / (self.S * self.sigma * sqrt_T)
        vega = self.S * sqrt_T * N_prime_d1
        
        # Delta
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1
        
        # Theta
        theta_common = -(self.S * N_prime_d1 * self.sigma) / (2 * sqrt_T)
        theta_call = theta_common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        theta_put = theta_common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        # Rho
        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return {
            'call': {
                'delta': delta_call, 'gamma': gamma, 'vega': vega,
                'theta': theta_call, 'rho': rho_call
            },
            'put': {
                'delta': delta_put, 'gamma': gamma, 'vega': vega,
                'theta': theta_put, 'rho': rho_put
            }
        }

class OptionVisualizer:
    """
    Class designed to visualize Option sensitivities and price movements.
    """
    
    @staticmethod
    def plot_greeks_surface(base_pricer: OptionPricer, 
                            Greek: str = 'Delta', 
                            OptionType: str = 'call',
                            spot_range: Tuple[float, float] = (0.8, 1.2), # relative to S
                            time_range: Tuple[float, float] = (0.01, 1.0)):
        """
        Generates a 3D Surface Plot of a specific Greek vs Spot Price and Time to Maturity.
        """
        S_values = np.linspace(base_pricer.S * spot_range[0], base_pricer.S * spot_range[1], 50)
        T_values = np.linspace(time_range[0], time_range[1], 50)
        
        S_grid, T_grid = np.meshgrid(S_values, T_values)
        Z = np.zeros_like(S_grid)
        
        # Recalculate Greek for each point in mesh
        for i in range(len(T_values)):
            for j in range(len(S_values)):
                temp_pricer = OptionPricer(
                    S=S_grid[i, j], 
                    K=base_pricer.K, 
                    T=T_grid[i, j], 
                    r=base_pricer.r, 
                    sigma=base_pricer.sigma
                )
                Z[i, j] = temp_pricer.calculate_greeks()[OptionType][Greek.lower()]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(S_grid, T_grid, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax.set_title(f'{OptionType.capitalize()} Option {Greek} Surface', fontsize=16, pad=20)
        ax.set_xlabel('Spot Price ($)', fontsize=12)
        ax.set_ylabel('Time to Maturity (Years)', fontsize=12)
        ax.set_zlabel(f'{Greek}', fontsize=12)
        
        fig.colorbar(surf, shrink=0.5, aspect=10, label=f'{Greek} Value')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap(base_pricer: OptionPricer,
                     Greek: str = 'Gamma',
                     OptionType: str = 'call',
                     spot_range: Tuple[float, float] = (0.8, 1.2),
                     time_range: Tuple[float, float] = (0.01, 1.0)):
        """
        Generates a Heatmap of a specific Greek vs Spot Price and Time to Maturity.
        """
        S_values = np.linspace(base_pricer.S * spot_range[0], base_pricer.S * spot_range[1], 50)
        T_values = np.linspace(time_range[0], time_range[1], 50)
        
        grid_data = np.zeros((len(T_values), len(S_values)))
        
        for i, t_val in enumerate(T_values):
            for j, s_val in enumerate(S_values):
                temp_pricer = OptionPricer(
                    S=s_val,
                    K=base_pricer.K,
                    T=t_val,
                    r=base_pricer.r,
                    sigma=base_pricer.sigma
                )
                grid_data[i, j] = temp_pricer.calculate_greeks()[OptionType][Greek.lower()]
        
        # Heatmap (inverted Y axis for intuitive Time logic if needed, but standard is fine)
        # Using T on Y-axis (increasing upwards) and S on X-axis
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(grid_data, xticklabels=10, yticklabels=10, cmap='coolwarm', annot=False)
        
        # Formatting ticks to show actual values
        ax.set_xticks(np.linspace(0, len(S_values)-1, 10))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(S_values[0], S_values[-1], 10)])
        
        ax.set_yticks(np.linspace(0, len(T_values)-1, 10))
        ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(T_values[0], T_values[-1], 10)])
        ax.invert_yaxis() # Put 0 time at bottom usually? Or T at top? Matplotlib 0 is bottom. Heatmap 0 is top. Inverting to have low T at bottom.
        
        plt.title(f'{OptionType.capitalize()} Option {Greek} Heatmap', fontsize=16, pad=20)
        plt.xlabel('Spot Price ($)', fontsize=12)
        plt.ylabel('Time to Maturity (Years)', fontsize=12)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # --- Demonstacja Programu ---
    
    # Parametry wejściowe
    S_spot = 100    # Cena spot
    K_strike = 100  # Cena wykonania
    T_time = 1.0    # Czas do wygaśnięcia (1 rok)
    r_rate = 0.05   # Stopa wolna od ryzyka (5%)
    sigma_vol = 0.2 # Zmienność (20%)
    
    print(f"--- Wycena Opcji BSM ---")
    print(f"Parametry: S={S_spot}, K={K_strike}, T={T_time}, r={r_rate}, sigma={sigma_vol}")
    
    # 1. Inicjalizacja i Obliczenia
    pricer = OptionPricer(S=S_spot, K=K_strike, T=T_time, r=r_rate, sigma=sigma_vol)
    prices = pricer.calculate_price()
    greeks = pricer.calculate_greeks()
    
    # 2. Wyświetlenie wyników tekstowych
    print(f"\nCeny Opcji:")
    print(f"Call Price: {prices['call']:.4f}")
    print(f"Put Price:  {prices['put']:.4f}")
    
    print(f"\nGreki (Call):")
    for delta_key, val in greeks['call'].items():
        print(f"{delta_key.capitalize()}: {val:.4f}")

    print(f"\nGreki (Put):")
    for delta_key, val in greeks['put'].items():
        print(f"{delta_key.capitalize()}: {val:.4f}")
        
    # 3. Wizualizacja
    print("\nGenerowanie wykresów...")
    
    # Przykład A: Surface Plot dla Delty (Call)
    OptionVisualizer.plot_greeks_surface(pricer, Greek='Delta', OptionType='call')
    
    # Przykład B: Heatmap dla Gammy (Call) - Gamma jest taka sama dla Put
    OptionVisualizer.plot_heatmap(pricer, Greek='Gamma', OptionType='call')
    
    print("Zakończono.")
