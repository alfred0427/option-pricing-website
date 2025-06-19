# 🧮 Black-Scholes 選擇權定價熱圖工具

## 📘 前言

這學期在課堂上學習了衍生性金融商品的定價，特別是經典的 Black-Scholes 模型。在學習過程中，我發現雖然可以套公式計算價格，但對「各個變數（如波動率、標的價格）」如何影響期權價格仍不夠直覺。因此，我使用 Python 和 Streamlit 製作了這個小工具，讓自己可以**用互動方式觀察參數變化對期權價格的影響**，也希望能幫助有類似困惑的同學。

---

## 📐 Black-Scholes 模型介紹

Black-Scholes 模型是一種用於歐式選擇權（European Options）的定價公式。其基本假設包括市場有效、資產報酬為常態分布、無套利等等。模型提供了在無風險利率 \( r \)、波動率 \( \sigma \)、剩餘到期時間 \( T \) 等條件下，選擇權的理論價格。

### 📄 公式如下：

> ![example](./bsmodel.png)


其中：
> ![example](./bs變數介紹.png)


---

## 💻 Code 架構說明

本專案主要分為兩個模組功能：

### 1. 📊 後端定價模組（`Option_pricing` 類別）

此類別負責：
- 建立不同 Spot 與 Volatility 的網格（Grid）
- 利用 Black-Scholes 模型計算對應的期權價格
- 計算結果儲存成 2D 陣列供繪圖使用

主要方法：
- `black_scholes()`：單點定價公式  
- `_calculate_prices()`：全網格計算價格矩陣  
- `plot_heatmap()`：繪製價格熱圖（Spot vs Volatility）

---

### 2. 🖼️ Streamlit 使用者介面（`st.sidebar.form` + 主畫面）

- 使用者可在左側欄位輸入參數（S, K, σ, r, T, 選擇權類型）
- 點選「✅ 生成熱圖」後會即時更新熱圖與中心價格顯示
- 圖中會特別用紅色框線標記出使用者輸入的那一組 S 與 σ 的價格位置

---

## 📷 展示畫面

> 以下是一張實際生成的熱圖（不同 Spot 和 Volatility 對 Call 選擇權價格的影響）  
> ![example](./image.png)

---
