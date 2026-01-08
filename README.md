# 📉 Black-Scholes Option Pricing & Greeks Visualization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SciPy](https://img.shields.io/badge/SciPy-Stats-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Status](https://img.shields.io/badge/Status-Educational-yellow)

## 📋 O Projekcie

Ten projekt to implementacja modelu wyceny opcji europejskich **Blacka-Scholesa-Mertona (BSM)** w języku Python.

Celem projektu jest demonstracja umiejętności z zakresu **Quantitative Development**: od implementacji matematycznych wzorów na wycenę instrumentów pochodnych, przez obliczanie wrażliwości (tzw. "Greeks"), aż po zaawansowaną wizualizację danych finansowych w 3D.

Kod został napisany w paradygmacie obiektowym (OOP), wykorzystując biblioteki naukowe `scipy` oraz `numpy` do wydajnych obliczeń numerycznych.

---

## 🧮 Matematyka i Model

Silnik wyceny opiera się na rozwiązaniu równania różniczkowego cząstkowego Blacka-Scholesa. Dla opcji europejskiej typu Call, cena $C(S,t)$ wyrażona jest wzorem:

$$C(S, t) = N(d_1)S_t - N(d_2)Ke^{-r(T-t)}$$

Gdzie:
* $S_t$ – Cena instrumentu bazowego (Spot Price)
* $K$ – Cena wykonania (Strike Price)
* $r$ – Stopa wolna od ryzyka (Risk-free rate)
* $\sigma$ – Zmienność (Volatility)
* $T-t$ – Czas do wygasnięcia

Parametry $d_1$ i $d_2$ obliczane są jako:

$$d_1 = \frac{\ln(S_t/K) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}}$$
$$d_2 = d_1 - \sigma\sqrt{T-t}$$

### Obliczane Greki (The Greeks)
System oblicza analitycznie pochodne cząstkowe, kluczowe dla zarządzania ryzykiem portfela:

* **Delta ($\Delta$):** Wrażliwość ceny opcji na zmianę ceny instrumentu bazowego.
* **Gamma ($\Gamma$):** Zmiana Delty względem ceny instrumentu bazowego (wypukłość).
* **Vega ($\nu$):** Wrażliwość na zmianę zmienności (Volatility).
* **Theta ($\Theta$):** Utrata wartości w czasie (Time Decay).
* **Rho ($\rho$):** Wrażliwość na zmianę stopy procentowej.

---

## 🚀 Funkcjonalności

1.  **Klasa `OptionPricer`:**
    * Hermetyzacja parametrów rynkowych.
    * Wykorzystanie `scipy.stats.norm` do precyzyjnego obliczania dystrybuanty (CDF) i funkcji gęstości (PDF) rozkładu normalnego.
    * Obsługa opcji typu Call i Put.

2.  **Klasa `OptionVisualizer`:**
    * Generowanie wykresów **3D Surface Plots** (np. Delta w funkcji Ceny i Czasu).
    * Generowanie map ciepła (**Heatmaps**) dla wizualizacji ryzyka Gamma.
    * Estetyczna stylizacja wykresów przy użyciu `seaborn` i `matplotlib`.

---

## 📊 Przykładowe Wizualizacje

*(Tutaj umieść screenshoty wygenerowane przez program. Przykłady poniżej)*

### 1. Powierzchnia Delty (Delta Surface)
Wizualizacja pokazująca, jak Delta opcji Call dąży do 1.0 (ITM) lub 0.0 (OTM) w miarę zbliżania się do wygaśnięcia.

![Delta Surface](https://via.placeholder.com/800x400?text=Place+Your+Delta+3D+Plot+Here)

### 2. Mapa Ciepła Gammy (Gamma Heatmap)
Obrazuje ryzyko zmiany Delty. Najwyższa Gamma występuje dla opcji "At The Money" tuż przed wygaśnięciem.

![Gamma Heatmap](https://via.placeholder.com/800x400?text=Place+Your+Gamma+Heatmap+Here)

---

## 🛠️ Instalacja i Użycie

### Wymagania
* Python 3.8+
* Biblioteki: `numpy`, `scipy`, `matplotlib`, `seaborn`

```bash
pip install numpy scipy matplotlib seaborn
