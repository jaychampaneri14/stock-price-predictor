"""
Stock Price Predictor — LSTM with technical indicators
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False
    print("PyTorch not available — using sklearn fallback")

# Generate synthetic stock data for demo
def generate_stock_data(n=1000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 100.0
    prices = []
    for _ in range(n):
        price *= (1 + np.random.normal(0.0005, 0.02))
        prices.append(price)
    df = pd.DataFrame({"Date": dates, "Close": prices})
    df["Volume"] = np.random.randint(1000000, 5000000, n)
    df["Open"] = df["Close"] * (1 + np.random.normal(0, 0.005, n))
    df["High"] = df[["Close", "Open"]].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
    df["Low"] = df[["Close", "Open"]].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
    return df.set_index("Date")

def add_technical_indicators(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["BB_upper"] = df["MA20"] + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["MA20"] - 2 * df["Close"].rolling(20).std()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    return df.dropna()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # predict Close
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

def train_lstm(X_train, y_train, epochs=30):
    model = LSTMModel(X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train).unsqueeze(1)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(Xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")
    return model

def main():
    print("=" * 60)
    print("  Stock Price Predictor (LSTM)")
    print("=" * 60)

    df = generate_stock_data()
    df = add_technical_indicators(df)
    print(f"Data: {len(df)} trading days with {len(df.columns)} features")

    features = ["Close", "Volume", "MA20", "MA50", "RSI", "Volatility"]
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    split = int(len(data_scaled) * 0.8)
    X, y = create_sequences(data_scaled, seq_len=60)
    X_train, X_test = X[:split-60], X[split-60:]
    y_train, y_test = y[:split-60], y[split-60:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print("Training LSTM...")

    if TORCH:
        model = train_lstm(X_train, y_train, epochs=30)
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test)).numpy().flatten()
    else:
        from sklearn.linear_model import Ridge
        preds = Ridge().fit(X_train.reshape(len(X_train),-1), y_train).predict(X_test.reshape(len(X_test),-1))

    # Inverse transform (only Close column)
    dummy = np.zeros((len(preds), data_scaled.shape[1]))
    dummy[:, 0] = preds
    preds_inv = scaler.inverse_transform(dummy)[:, 0]
    dummy2 = np.zeros((len(y_test), data_scaled.shape[1]))
    dummy2[:, 0] = y_test
    actual = scaler.inverse_transform(dummy2)[:, 0]

    rmse = mean_squared_error(actual, preds_inv, squared=False)
    mae = mean_absolute_error(actual, preds_inv)
    print(f"\nTest RMSE: {rmse:.2f}")
    print(f"Test MAE:  {mae:.2f}")
    print(f"Next day predicted price: ${preds_inv[-1]:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", alpha=0.8)
    plt.plot(preds_inv, label="Predicted", alpha=0.8)
    plt.title("Stock Price Prediction (LSTM)")
    plt.legend(); plt.tight_layout()
    plt.savefig("stock_prediction.png", dpi=100)
    print("Chart saved: stock_prediction.png")

if __name__ == "__main__":
    main()
