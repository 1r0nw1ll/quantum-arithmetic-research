import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import io
from tqdm import tqdm
import yfinance as yf

# --- 1. Core QA-Markovian Engine (Verified) ---
MODULUS = 24
E8_ROOTS = None

def get_qa_tuples_vectorized(b, e, mod=MODULUS):
    d = (b + e) % mod; a = (b + 2 * e) % mod
    return np.stack([b, e, d, a], axis=1)

def get_node_harmonic_loss(b, e, d, a, mod=MODULUS):
    lhs = np.mod(a**2, mod); rhs = np.mod(d**2 + 2*d*e + e**2, mod)
    diff = np.abs(lhs - rhs); diff = np.minimum(diff, mod - diff)
    return diff**2

def qa_inner_product_matrix(tuples, mod=MODULUS):
    return np.mod(np.einsum('ni,mi->nm', tuples.astype(int), tuples.astype(int)), mod)

def generate_e8_root_system():
    global E8_ROOTS
    if E8_ROOTS is not None: return E8_ROOTS
    roots = set()
    for i, j in itertools.combinations(range(8), 2):
        for s1, s2 in itertools.product([-1, 1], repeat=2):
            v = np.zeros(8); v[i], v[j] = s1, s2; roots.add(tuple(v))
    for signs in itertools.product([-0.5, 0.5], repeat=8):
        if np.sum(signs) % 1 == 0: roots.add(signs)
    E8_ROOTS = np.array(list(roots))
    return E8_ROOTS

def analyze_signal_window(signal_window):
    NUM_NODES = 24; TIMESTEPS_INTERNAL = 75
    COUPLING = 0.1; NOISE_STRENGTH = 0.2; NOISE_ANNEALING = 0.97
    INJECTION_STRENGTH = 0.1
    b_state = np.random.uniform(0, MODULUS, NUM_NODES)
    e_state = np.random.uniform(0, MODULUS, NUM_NODES)
    injection_values = signal_window * INJECTION_STRENGTH
    for t in range(TIMESTEPS_INTERNAL):
        injection_value = injection_values[t % len(injection_values)]
        current_tuples = get_qa_tuples_vectorized(b_state, e_state)
        resonance = qa_inner_product_matrix(current_tuples)
        weights = resonance / (np.sum(resonance, axis=1, keepdims=True) + 1e-9)
        current_be_state = np.stack([b_state, e_state], axis=1)
        neighbor_pull = weights @ current_be_state
        noise = (np.random.rand(NUM_NODES, 2) - 0.5) * NOISE_STRENGTH * (NOISE_ANNEALING ** t)
        delta = COUPLING * (neighbor_pull - current_be_state) + noise
        b_state = np.mod(b_state + delta[:, 0] + injection_value, MODULUS)
        e_state = np.mod(e_state + delta[:, 1], MODULUS)
    final_tuples_4d = get_qa_tuples_vectorized(b_state, e_state)
    final_b, final_e, final_d, final_a = final_tuples_4d.T
    final_loss = np.mean(get_node_harmonic_loss(final_b, final_e, final_d, final_a))
    final_tuples_8d = np.zeros((NUM_NODES, 8)); final_tuples_8d[:, :4] = final_tuples_4d
    norms_state = np.linalg.norm(final_tuples_8d, axis=1, keepdims=True)
    normalized_states = final_tuples_8d / (norms_state + 1e-9)
    cosine_similarities = np.abs(normalized_states @ E8_ROOTS_NORMALIZED.T)
    mean_e8_alignment = np.mean(np.max(cosine_similarities, axis=1))
    return mean_e8_alignment * np.exp(-0.1 * final_loss)

# --- 2. Performance Metrics Calculation ---
def calculate_performance_metrics(equity_curve):
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    rolling_max = equity_curve.cummax()
    daily_drawdown = equity_curve / rolling_max - 1.0
    max_drawdown = daily_drawdown.min() * 100
    return {
        "Total Return (%)": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown
    }

# --- 3. Main Backtesting Execution ---
if __name__ == "__main__":
    print("--- Advanced Backtest on REAL DATA: Harmonic Trend Confirmation ---")
    
    e8_roots = generate_e8_root_system()
    norms_e8 = np.linalg.norm(e8_roots, axis=1, keepdims=True)
    E8_ROOTS_NORMALIZED = e8_roots / (norms_e8 + 1e-9)

    # Step 1: Load Data
    print("Step 1: Loading real-world SPY price history...")
    # Download SPY data using yfinance
    spy_data = yf.download("SPY", start="2000-01-01", end="2023-12-31")
    spy = pd.DataFrame({'Price': spy_data['Close', 'SPY']})
    spy['Returns'] = spy['Price'].pct_change().fillna(0)

    # Step 2: Generate Harmonic Index and Technical Indicators
    print("Step 2: Generating Harmonic Index and technical indicators...")
    rolling_window = 252
    hi_history = []
    # Adjust the loop range to account for the rolling window and monthly calculation
    # Ensure there's enough data for the rolling window and at least one month (21 trading days)
    start_index = rolling_window + 21 # Minimum data needed for first HI calculation
    for i in tqdm(range(start_index, len(spy), 21)): # Calculate monthly
        window_returns = spy['Returns'].iloc[i-rolling_window : i]
        # Ensure window_returns is not empty and has sufficient variance
        if len(window_returns) == 0 or window_returns.std() == 0:
            hi_score = 0.0 # Assign a default HI score if data is insufficient
        else:
            normalized_window = (window_returns - window_returns.mean()) / (window_returns.std() + 1e-9)
            hi_score = analyze_signal_window(normalized_window.values)
        hi_history.append({'Date': spy.index[i], 'HarmonicIndex': hi_score})
    hi_df = pd.DataFrame(hi_history).set_index('Date')
    
    # --- NEW: Add Technical Indicators ---
    df = spy.join(hi_df).fillna(method='ffill').dropna()
    df['SMA200'] = df['Price'].rolling(window=200).mean()
    df['HI_RoC'] = df['HarmonicIndex'].pct_change(periods=3) # 3-month Rate of Change

    df.dropna(inplace=True)
    
    # Step 3: Apply Advanced Strategy Rules
    print("Step 3: Applying advanced trading rules...")
    
    # Condition 1: Price is below the 200-day SMA
    trend_exit = df['Price'] < df['SMA200']
    
    # Condition 2: Harmonic Index Rate of Change collapses
    harmonic_exit = df['HI_RoC'] < -0.5
    
    # The signal is 0 (Risk-Off) if EITHER exit condition is met
    df['Signal'] = np.where(trend_exit & harmonic_exit, 0, 1)
    
    # Shift signal to avoid lookahead bias
    df['Signal'] = df['Signal'].shift(1).fillna(0)
    
    # --- Calculate Returns and Equity Curves ---
    df['BuyAndHold_Returns'] = df['Returns']
    df['HarmonicStrategy_Returns'] = df['Returns'] * df['Signal']
    df['BuyAndHold_Equity'] = (1 + df['BuyAndHold_Returns']).cumprod()
    df['AdvancedStrategy_Equity'] = (1 + df['HarmonicStrategy_Returns']).cumprod()

    # Step 4: Calculate Performance
    print("Step 4: Calculating final performance...")
    buy_and_hold_metrics = calculate_performance_metrics(df['BuyAndHold_Equity'])
    advanced_strategy_metrics = calculate_performance_metrics(df['AdvancedStrategy_Equity'])
    results_df = pd.DataFrame([buy_and_hold_metrics, advanced_strategy_metrics],
                              index=["Buy and Hold", "Advanced Harmonic Strategy"])
    
    print("\n" + "="*60)
    print("--- ADVANCED Backtest Results (2000-2023) ---")
    print(results_df)
    print("="*60 + "\n")

    # Step 5: Visualization
    print("Step 5: Generating visualizations...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 3]})
    
    ax1.plot(df.index, df['Price'], color='black', label='S&P 500 (SPY)')
    ax1.plot(df.index, df['SMA200'], color='orange', linestyle='--', label='200-Day SMA')
    ax1.set_title('S&P 500 with Advanced Harmonic Regimes', fontsize=16)
    ax1.set_ylabel('Price (log scale)'); ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    ax1.fill_between(df.index, df['Price'].min(), df['Price'].max(), 
                     where=df['Signal']==0, color='red', alpha=0.2, label='Risk-Off Regime')
    ax1.legend()

    ax2.plot(df.index, df['BuyAndHold_Equity'], color='gray', linestyle='--', label='Buy and Hold')
    ax2.plot(df.index, df['AdvancedStrategy_Equity'], color='purple', label='Advanced Harmonic Strategy')
    ax2.set_title('Advanced Strategy Performance', fontsize=16)
    ax2.set_ylabel('Portfolio Growth ($1 initial investment)'); ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.suptitle('Advanced Strategy: Harmonic Trend Confirmation', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
