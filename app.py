import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

class Option_pricing():
    def __init__(self, K ,S, vol, T=1, r=0, option_type="call", grid_size=9):
        self.S = S
        self.K = K 
        self.T = T
        self.r = r
        self.vol = vol
        self.option_type = option_type
        self.grid_size = grid_size

        self.spot_prices = np.linspace(max(self.S - (self.vol * self.S), 0),
                                       self.S + (self.vol * self.S),
                                       self.grid_size)
        self.volatilities = np.linspace(0.5 * self.vol, 1.5 * self.vol, self.grid_size)

        self.option_prices = np.zeros((len(self.spot_prices), len(self.volatilities)))
        self._calculate_prices()

    def black_scholes(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

    def _calculate_prices(self):
        for i, spot in enumerate(self.spot_prices):
            for j, sigma in enumerate(self.volatilities):
                self.option_prices[i, j] = self.black_scholes(
                    spot, self.K, self.T, self.r, sigma, self.option_type
                )

    def plot_heatmap(self):
        fig, ax = plt.subplots(figsize=(12.5, 10))
        # 轉置價格矩陣（X 軸：Spot，Y 軸：Vol）
        im = ax.imshow(self.option_prices.T, origin='lower', cmap='viridis', aspect='auto')

        ax.set_xticks(np.arange(len(self.spot_prices)))
        ax.set_yticks(np.arange(len(self.volatilities)))
        ax.set_xticklabels([f"{s:.2f}" for s in self.spot_prices])
        ax.set_yticklabels([f"{v:.3f}" for v in self.volatilities])

        ax.set_xticks(np.arange(len(self.spot_prices) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(self.volatilities) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        center_i = len(self.spot_prices) // 2
        center_j = len(self.volatilities) // 2
        for i in range(len(self.spot_prices)):
            for j in range(len(self.volatilities)):
                val = self.option_prices[i, j]
                color = "black" if val > self.option_prices.max() * 0.5 else "white"
                # 注意！此處畫圖要對應轉置座標 (i, j) → (i, j)
                if i == center_i and j == center_j:
                    ax.text(i, j, f"{val:.2f}",
                            ha='center', va='center', color='red', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
                else:
                    ax.text(i, j, f"{val:.2f}", ha='center', va='center',
                            color=color, fontsize=9, fontweight='bold')

        fig.colorbar(im, ax=ax, label=f"{self.option_type.capitalize()} Option Price")
        ax.set_title(f"{self.option_type.capitalize()} Option Price Heatmap")
        ax.set_xlabel("Spot Price (S)")
        ax.set_ylabel("Volatility (σ)")
        return fig






# === Streamlit UI ===

st.set_page_config(
    page_title="Black-Scholes 定價工具",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("Black-Scholes 選擇權定價熱圖工具")

st.markdown("""
這是一個使用 **Black-Scholes 模型** 製作的選擇權定價工具。
您可以在左側輸入參數（如標的價格、履約價、波動率等），
按下下方的「✅執行」按鈕後，即可在主畫面中看到對應的定價以及熱度圖。


- 📈 價格熱圖：顯示在不同標的價格(S)與波動率(σ)下的模型定價  
- 📍 熱度圖中心點：即為您輸入的 S 與 σ 所對應的模型價格  
""")



# Sidebar with form
with st.sidebar.form("option_form"):
    K = st.number_input("Strike Price (K)", value=100.0)
    S = st.number_input("Spot Price (S)", value=100.0)
    vol = st.number_input("Volatility (σ)", value=0.2, min_value=0.01)
    T = st.number_input("Time to Maturity (T, 年)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    option_type = st.selectbox("Option Type", ("call", "put"))
    grid_size = st.slider("Grid Size", 3, 15, 9, step=2)
    
    submitted = st.form_submit_button("✅ 生成熱圖")
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 作者：清大計財系 陳冠熏")


# Only run this when the button is clicked
if submitted:
    op = Option_pricing(K=K, S=S, vol=vol, T=T, r=r, option_type=option_type, grid_size=grid_size)
    center_price = op.option_prices[grid_size // 2, grid_size // 2]
    st.subheader(f"🎯 選擇權理論價格: (S={S}, σ={vol}): `{center_price:.4f}`")
    fig = op.plot_heatmap()
    st.pyplot(fig, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("👨‍💻 作者：清大計財系 陳冠熏", unsafe_allow_html=True)

