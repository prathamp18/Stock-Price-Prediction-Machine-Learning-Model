# 📈 Stock Price Prediction using LSTM (NVIDIA Example)

This project demonstrates how to use a **Long Short-Term Memory (LSTM)** neural network to predict future stock prices using historical stock market data.  
The model learns patterns from previous days' prices to forecast the **next day’s closing price**.

---

## 🚀 Features

- Fetches live historical stock data via **Yahoo Finance**
- Scales and prepares time-series data
- Builds and trains a **stacked LSTM** neural network
- Visualizes actual vs predicted stock prices
- Predicts the **next trading day’s** closing price

---

## 🧠 Model Architecture

| Layer | Type | Units | Details |
|-------|------|--------|----------|
| 1 | LSTM | 100 | `return_sequences=True` |
| 2 | Dropout | 0.2 | Prevents overfitting |
| 3 | LSTM | 100 | `return_sequences=True` |
| 4 | Dropout | 0.2 | Regularization |
| 5 | LSTM | 100 | — |
| 6 | Dropout | 0.2 | Regularization |
| 7 | Dense | 1 | Final output layer (Predicted price) |

- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error  
- **Epochs:** 50  
- **Batch Size:** 32  

---
