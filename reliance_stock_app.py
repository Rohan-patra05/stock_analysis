# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import joblib
# import datetime
# import matplotlib.pyplot as plt

# # Load model and scaler
# model = joblib.load("reliance_model.pkl")
# scaler = joblib.load("reliance_scaler.pkl")

# st.title("📈 Stock Price Prediction App")
# st.markdown("Predict the **next trading day's closing price** using historical market data.")

# # Sidebar for inputs
# st.sidebar.header("📅 Data Settings")

# ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", value="")
# if not ticker:
#     st.warning("Please enter a stock ticker to proceed.")
#     st.stop()

# start_date = st.sidebar.date_input("Select Start Date")
# end_date = st.sidebar.date_input("Select End Date", value=datetime.date.today())

# # Ensure valid date range
# if start_date >= end_date:
#     st.error("⚠️ Start date must be before end date.")
#     st.stop()

# # Download stock data
# st.subheader(f"Historical Data for {ticker}")
# df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# if df.empty:
#     st.error("No data found for this ticker/date range.")
#     st.stop()

# st.write(df.tail())

# # Plot closing price
# st.subheader("Closing Price Trend")
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(df["Close"], label="Closing Price")
# ax.set_xlabel("Date")
# ax.set_ylabel("Price (INR)")
# ax.legend()
# st.pyplot(fig)

# # Prepare latest data for prediction
# latest_data = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
# latest_scaled = scaler.transform(latest_data)
# predicted_price = model.predict(latest_scaled)[0].item()

# st.subheader("Prediction")
# st.write(f"**Predicted closing price for the next trading day:** ₹{predicted_price:,.2f}")

# # Manual prediction input
# st.sidebar.header("✏️ Manual Prediction Input")
# open_price = st.sidebar.number_input("Open Price", value=float(df['Open'].iloc[-1]))
# high_price = st.sidebar.number_input("High Price", value=float(df['High'].iloc[-1]))
# low_price = st.sidebar.number_input("Low Price", value=float(df['Low'].iloc[-1]))
# volume = st.sidebar.number_input("Volume", value=float(df['Volume'].iloc[-1]))

# if st.sidebar.button("Predict from Manual Input"):
#     manual_data = np.array([[open_price, high_price, low_price, volume]])
#     manual_scaled = scaler.transform(manual_data)
#     manual_pred = model.predict(manual_scaled)[0].item()
#     st.write(f"Manual Input Prediction: ₹{manual_pred:,.2f}")

# st.success("✅ App is ready. Use the sidebar to enter your stock and date range.")






'''
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("reliance_model.pkl")
scaler = joblib.load("reliance_scaler.pkl")

st.title("📈 Stock Price Prediction App")
st.markdown("Predict the **next trading day's closing price** using historical market data.")

# Sidebar for inputs
st.sidebar.header("📅 Data Settings")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", value="")
if not ticker:
    st.warning("Please enter a stock ticker to proceed.")
    st.stop()

start_date = st.sidebar.date_input("Select Start Date")
end_date = st.sidebar.date_input("Select End Date", value=datetime.date.today())

# Ensure valid date range
if start_date >= end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()

# Download stock data
st.subheader(f"Historical Data for {ticker}")
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

if df.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

st.write(df.tail())

# Calculate moving averages
df["MA50"] = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

# Plot closing price + moving averages
st.subheader("Closing Price with Moving Averages")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Close"], label="Closing Price", color='blue')
ax.plot(df["MA50"], label="50-Day MA", color='red', linestyle='--')
ax.plot(df["MA200"], label="200-Day MA", color='green', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.set_title("Closing Price with 50 & 200-Day Moving Averages")
ax.legend()
st.pyplot(fig)

# Trading Volume Over Time
st.subheader("Trading Volume Over Time")
fig_vol, ax_vol = plt.subplots(figsize=(10, 5))
ax_vol.plot(df["Volume"], label="Volume", color='orange')
ax_vol.set_xlabel("Date")
ax_vol.set_ylabel("Shares Traded")
ax_vol.set_title("Trading Volume Over Time")
ax_vol.legend()
st.pyplot(fig_vol)

# Distribution of Closing Prices
st.subheader("Distribution of Closing Prices")
fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
ax_hist.hist(df["Close"], bins=30, color='green', alpha=0.7)
ax_hist.set_xlabel("Closing Price (INR)")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("Closing Price Distribution")
st.pyplot(fig_hist)

# Open vs Close Price Scatter Plot
st.subheader("Open vs Close Price Relationship")
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
ax_scatter.scatter(df["Open"], df["Close"], alpha=0.6, color='purple')
ax_scatter.set_xlabel("Opening Price (INR)")
ax_scatter.set_ylabel("Closing Price (INR)")
ax_scatter.set_title("Open vs Close Price Scatter Plot")
st.pyplot(fig_scatter)

# Prepare latest data for prediction
latest_data = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_data)
predicted_price = model.predict(latest_scaled)[0].item()

st.subheader("Prediction")
st.write(f"**Predicted closing price for the next trading day:** ₹{predicted_price:,.2f}")

# Manual prediction input
st.sidebar.header("✏️ Manual Prediction Input")
open_price = st.sidebar.number_input("Open Price", value=float(df['Open'].iloc[-1]))
high_price = st.sidebar.number_input("High Price", value=float(df['High'].iloc[-1]))
low_price = st.sidebar.number_input("Low Price", value=float(df['Low'].iloc[-1]))
volume = st.sidebar.number_input("Volume", value=float(df['Volume'].iloc[-1]))

if st.sidebar.button("Predict from Manual Input"):
    manual_data = np.array([[open_price, high_price, low_price, volume]])
    manual_scaled = scaler.transform(manual_data)
    manual_pred = model.predict(manual_scaled)[0].item()
    st.write(f"Manual Input Prediction: ₹{manual_pred:,.2f}")

st.success("✅ App is ready. Use the sidebar to enter your stock and date range.")
'''








# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("reliance_model.pkl")
scaler = joblib.load("reliance_scaler.pkl")

st.title("📈 Stock Price Prediction App")
st.markdown("Predict the **next trading day's closing price** using historical market data.")

# Sidebar for inputs
st.sidebar.header("📅 Data Settings")

# Dropdown for stock selection
stock_options = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Jio Financial Services": "JIOFIN.NS",
    "IRCTC": "IRCTC.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS"
}

selected_name = st.sidebar.selectbox("Select a Stock", list(stock_options.keys()))
ticker_symbol = stock_options[selected_name]

start_date = st.sidebar.date_input("Select Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("Select End Date", value=datetime.date.today())

# Ensure valid date range
if start_date >= end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()

# Download stock data
st.subheader(f"Historical Data for {selected_name}")
df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)

if df.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

st.write(df.tail())

# Calculate moving averages
df["MA50"] = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

# Closing Price + MA
st.subheader("Closing Price with Moving Averages")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Close"], label="Closing Price", color='blue')
ax.plot(df["MA50"], label="50-Day MA", color='red', linestyle='--')
ax.plot(df["MA200"], label="200-Day MA", color='green', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.set_title("Closing Price with 50 & 200-Day Moving Averages")
ax.legend()
st.pyplot(fig)

# Trading Volume
st.subheader("Trading Volume Over Time")
fig_vol, ax_vol = plt.subplots(figsize=(10, 5))
ax_vol.plot(df["Volume"], label="Volume", color='orange')
ax_vol.set_xlabel("Date")
ax_vol.set_ylabel("Shares Traded")
ax_vol.set_title("Trading Volume Over Time")
ax_vol.legend()
st.pyplot(fig_vol)

# Histogram
st.subheader("Distribution of Closing Prices")
fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
ax_hist.hist(df["Close"], bins=30, color='green', alpha=0.7)
ax_hist.set_xlabel("Closing Price (INR)")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("Closing Price Distribution")
st.pyplot(fig_hist)

# Scatter Plot
st.subheader("Open vs Close Price Relationship")
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
ax_scatter.scatter(df["Open"], df["Close"], alpha=0.6, color='purple')
ax_scatter.set_xlabel("Opening Price (INR)")
ax_scatter.set_ylabel("Closing Price (INR)")
ax_scatter.set_title("Open vs Close Price Scatter Plot")
st.pyplot(fig_scatter)

# RSI
st.subheader("Relative Strength Index (RSI)")
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
ax_rsi.plot(df["RSI"], label="RSI", color='brown')
ax_rsi.axhline(70, color='red', linestyle='--', label="Overbought")
ax_rsi.axhline(30, color='green', linestyle='--', label="Oversold")
ax_rsi.set_xlabel("Date")
ax_rsi.set_ylabel("RSI Value")
ax_rsi.set_title("Relative Strength Index (RSI)")
ax_rsi.legend()
st.pyplot(fig_rsi)

# MACD
st.subheader("MACD (Moving Average Convergence Divergence)")
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()

fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
ax_macd.plot(macd, label="MACD", color='blue')
ax_macd.plot(signal, label="Signal Line", color='red')
ax_macd.set_xlabel("Date")
ax_macd.set_ylabel("MACD Value")
ax_macd.set_title("MACD Indicator")
ax_macd.legend()
st.pyplot(fig_macd)

# Prediction
latest_data = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_data)
predicted_price = model.predict(latest_scaled)[0].item()

st.subheader("Prediction")
st.write(f"**Predicted closing price for the next trading day:** ₹{predicted_price:,.2f}")

# Manual Prediction
st.sidebar.header("✏️ Manual Prediction Input")
open_price = st.sidebar.number_input("Open Price", value=float(df['Open'].iloc[-1]))
high_price = st.sidebar.number_input("High Price", value=float(df['High'].iloc[-1]))
low_price = st.sidebar.number_input("Low Price", value=float(df['Low'].iloc[-1]))
volume = st.sidebar.number_input("Volume", value=float(df['Volume'].iloc[-1]))

if st.sidebar.button("Predict from Manual Input"):
    manual_data = np.array([[open_price, high_price, low_price, volume]])
    manual_scaled = scaler.transform(manual_data)
    manual_pred = model.predict(manual_scaled)[0].item()
    st.write(f"Manual Input Prediction: ₹{manual_pred:,.2f}")

st.success("✅ App is ready. Use the sidebar to select a stock and date range.")



# Execution:-  streamlit run reliance_stock_app.py


# # User input for a specific date
# st.sidebar.header("📅 Specific Date Target Calculation")
# year_input = st.sidebar.number_input("Year", min_value=1900, max_value=datetime.date.today().year, value=2025)
# month_input = st.sidebar.number_input("Month", min_value=1, max_value=12, value=8)
# day_input = st.sidebar.number_input("Day", min_value=1, max_value=31, value=10)

# if st.sidebar.button("Calculate Target for Date"):
#     try:
#         target_date = datetime.date(year_input, month_input, day_input)
        
#         # Find the nearest previous trading day
#         available_dates = df.index
#         if pd.Timestamp(target_date) not in available_dates:
#             nearest_date = available_dates[available_dates <= pd.Timestamp(target_date)].max()
#             if pd.isna(nearest_date):
#                 st.error(f"No trading data available before {target_date}")
#                 st.stop()
#             st.warning(f"No exact data for {target_date}. Using nearest previous trading day: {nearest_date.date()}")
#         else:
#             nearest_date = pd.Timestamp(target_date)
        
#         row = df.loc[nearest_date]
#         target_value = (row["Open"] + row["Close"]) / 2
#         st.write(f"**Target (Open + Close) / 2 for {nearest_date.date()}:** ₹{target_value:,.2f}")
        
#     except Exception as e:
#         st.error(f"Error: {e}")