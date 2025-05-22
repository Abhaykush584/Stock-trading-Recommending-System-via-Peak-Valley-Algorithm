import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Load data
df = pd.read_csv('features_aapl.csv')
df['Date'] = pd.to_datetime(df['Date'])


# Add features
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
df['Volatility'] = df['Return'].rolling(window=5).std()
df = df.dropna()
df.to_csv('features_enhanced_aapl.csv', index=False)
print(f"Saved {len(df)} rows to features_enhanced_aapl.csv")

# Train model
features = ['Close', 'MA5', 'MA10', 'Return', 'RSI', 'Volume_MA5', 'Volatility']
df['Target'] = df['Close'].shift(-1)
df = df.dropna()
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
X_train = train_df[features]
y_train = train_df['Target']
X_test = test_df[features]
y_test = test_df['Target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled feature means:", X_train_scaled.mean(axis=0))  # Should be ~0

param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train)

model = grid_search.best_estimator_
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
feature_importance = pd.Series(model.feature_importances_, index=features)

results = pd.DataFrame({
    'Date': test_df['Date'].reset_index(drop=True),
    'Actual_Close': y_test.reset_index(drop=True),
    'Predicted_Close': predictions
})
results['Date'] = pd.to_datetime(results['Date'])
results.to_csv('predictions_enhanced_aapl.csv', index=False)
print(f"Saved {len(results)} predictions to predictions_enhanced_aapl.csv")
print(f"Enhanced MAE: ${mae:.2f}")
print("Feature Importance:\n", feature_importance)

# Email alert function
def send_email_alert(trade, email_config):
    try:
        subject = f"Trade Alert: {'Buy' if trade['Profit'] >= 0 else 'Sell'} Signal"
        body = (f"Trade Recommendation:\n"
                f"Action: {'Buy' if trade['Profit'] >= 0 else 'Sell'}\n"
                f"Date: {trade['Buy_Date'] if trade['Profit'] >= 0 else trade['Sell_Date']}\n"
                f"Price: ${trade['Buy_Price'] if trade['Profit'] >= 0 else trade['Sell_Price']:.2f}\n"
                f"Confidence: {trade['Confidence']:.2f}\n"
                f"Portfolio Value: ${trade['Portfolio_Value']:.2f}")
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['receiver_email']
        
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
        print(f"Email sent for trade at {trade['Buy_Date']}")
    except Exception as e:
        print(f"Email error: {e}")

# Peak-valley
prices = results['Predicted_Close'].values
dates = results['Date']  # Use pd.Series of pd.Timestamp
cash = 10000
shares = 0
trades = []
portfolio = []
daily_trades = {}
print("Date type check:", type(dates.iloc[0]))  # Should be pd.Timestamp
i = 0
while i < len(prices) - 1:
    valley_start = i
    while i < len(prices) - 1 and prices[i] >= prices[i + 1] * 1.002:
        i += 1
    if i >= len(prices) - 1 or i - valley_start > 96:
        break
    buy_index = i
    buy_price = prices[i]
    buy_date = dates.iloc[i]
    
    buy_day = buy_date.strftime('%Y-%m-%d')
    daily_trades[buy_day] = daily_trades.get(buy_day, 0) + 1
    if daily_trades[buy_day] > 2:
        i += 1
        continue
    
    peak_start = i
    for j in range(i, min(i + 97, len(prices))):
        i = j
        current_price = prices[i]
        if current_price < buy_price * 0.98:
            sell_price = current_price
            sell_date = dates.iloc[j]
            break
        if j < len(prices) - 1 and prices[j] > prices[j + 1] * 1.002:
            sell_price = prices[j]
            sell_date = dates.iloc[j]
            break
    else:
        continue
    
    profit = sell_price - buy_price
    if profit > 0.10 or profit < -buy_price * 0.02:
        fee = (buy_price + sell_price) * 0.001
        net_profit = profit - fee
        confidence = min(1.0, (abs(profit) / buy_price) / (mae / buy_price))
        shares_bought = (cash * 0.5) // buy_price
        if shares_bought > 0:
            cash -= shares_bought * buy_price * 1.001
            shares += shares_bought
            cash += shares_bought * sell_price * 0.999
            shares -= shares_bought
            portfolio_value = cash + shares * sell_price
            trade = {
                'Buy_Date': buy_date,
                'Buy_Price': buy_price,
                'Sell_Date': sell_date,
                'Sell_Price': sell_price,
                'Profit': net_profit,
                'Time_Frame': 'hourly',
                'Confidence': confidence,
                'Portfolio_Value': portfolio_value
            }
            trades.append(trade)
            portfolio.append({
                'Date': sell_date,
                'Cash': cash,
                'Shares': shares,
                'Portfolio_Value': portfolio_value
            })
            email_config = {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'abhaykush050804@gmail.com',
                'sender_password': 'kckjuyxwyjycalye',
                'receiver_email': 'akks1925@gmail.com'
            }
            send_email_alert(trade, email_config)

trades_df = pd.DataFrame(trades)
portfolio_df = pd.DataFrame(portfolio)
if not trades_df.empty:
    trades_df.to_csv('trades_aapl.csv', index=False)
    portfolio_df.to_csv('portfolio_aapl.csv', index=False)
    print(f"Saved {len(trades_df)} trades to trades_aapl.csv")
    print(f"Saved {len(portfolio_df)} portfolio entries to portfolio_aapl.csv")

# Portfolio verdict
if not trades_df.empty:
    total_profit = trades_df['Profit'].sum()
    win_rate = (trades_df['Profit'] > 0).mean() * 100
    final_value = portfolio_df['Portfolio_Value'].iloc[-1]
    growth = ((final_value - 10000) / 10000) * 100
    cash_ratio = portfolio_df['Cash'].iloc[-1] / final_value
    diversification = "Balanced" if 0.4 <= cash_ratio <= 0.6 else "Unbalanced"
    returns = trades_df['Profit'] / trades_df['Buy_Price']
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)
    performance = "Strong" if growth > 2 else "Moderate" if growth > 0 else "Loss"
    print(f"Portfolio Verdict: Total Profit: ${total_profit:.2f}, "
          f"Win Rate: {win_rate:.1f}%, Growth: {growth:.2f}%, "
          f"Diversification: {diversification}, Sharpe Ratio: {sharpe_ratio:.2f}, "
          f"Performance: {performance}")
else:
    print("Portfolio Verdict: No trades executed.")