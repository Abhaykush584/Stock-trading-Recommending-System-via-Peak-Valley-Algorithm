# -End-to-End-stock-trading-system
Built an end-to-end stock trading system using AAPL stock data from an API.

📈 Stock Trading Recommendation System using Machine Learning
This project is a full-fledged stock trading strategy system built to predict short-term stock prices, identify optimal buy/sell points, and simulate portfolio performance using a combination of machine learning (Random Forest) and a Peak-Valley algorithm. The system is focused on Apple Inc. (AAPL) data and is designed for both learning and real-world portfolio simulation.

🚀 Project Workflow
✅ Step 1: Data Collection
Collected hourly stock data (AAPL) using a CSV from Yahoo Finance API or manual source.

Stored in cleaned_aapl.csv.

✅ Step 2: Data Cleaning
Removed nulls, handled time formatting, and ensured column consistency.

Output: Cleaned dataset for feature engineering.

✅ Step 3: Feature Engineering
Generated technical indicators:

MA5, MA10: 5/10-hour moving averages.

Return: Percentage price change per hour.

Output: features_aapl.csv (~1,638 rows).

✅ Step 4: Machine Learning Price Prediction
Model Used: Random Forest Regressor

Features: Close, MA5, MA10, Return

Target: Next hour’s Close

Accuracy: Evaluated using Mean Absolute Error (MAE)

Output: predictions_aapl.csv

✅ Step 5: Peak-Valley Algorithm
Applied custom logic to identify buy at valleys and sell at peaks in predicted prices.

Strategy simulates trading to maximize profit.

Output:

trades_aapl.csv (Trade details)

portfolio_aapl.csv (Portfolio simulation)

✅ Step 6: ML Enhancement + Email Alerts
Added more features: RSI, Volume MA5, Volatility

Applied StandardScaler and GridSearchCV for hyperparameter tuning.

Integrated Gmail alerts for real-time trade signals (requires app password).

Output:

features_enhanced_aapl.csv

predictions_enhanced_aapl.csv

Updated trades and portfolio CSVs

✅ Step 7: Streamlit Dashboard
Interactive visualization of:

Predicted vs Actual Prices

Buy/Sell signals

Portfolio growth

Performance metrics (profit, win rate, Sharpe Ratio, etc.)

Run via:

bash
Copy
Edit
streamlit run dashboard.py
📦 Folder Structure
Copy
Edit
.
├── cleaned_aapl.csv
├── features_aapl.csv
├── predictions_aapl.csv
├── features_enhanced_aapl.csv
├── predictions_enhanced_aapl.csv
├── trades_aapl.csv
├── portfolio_aapl.csv
├── ml_prediction.py
├── ml_enhancement.py
├── peak_valley.py
├── dashboard.py
└── README.md
🧠 Key Concepts & Integration
🔁 Return Column:
Not predicted by ML. Instead, it's a feature indicating momentum.

Helps Random Forest understand recent trends and improve accuracy.

📊 Random Forest:
Trained to predict future Close price using historical and technical data.

Core predictive engine that powers trading logic.

📈 Peak-Valley Strategy:
Core trading algorithm.

Detects price valleys (buy signals) and peaks (sell signals) based on predicted price movements.

Integrated with actual predicted data to simulate realistic trades.

📧 Email Alerts:
Configured with Gmail SMTP.

Sends alerts for each trade executed (buy/sell) based on peak-valley detection.

Customize in ml_enhancement.py.

📊 Sample Output
Predicted Price Accuracy: MAE ~$1.34

Profit from Trades: ~$8.20 in simulation

Sharpe Ratio: ~1.60 (indicative of healthy strategy)

Number of Trades: ~10–20 over test period

💻 How to Run
1. Install Dependencies
bash
Copy
Edit
pip install pandas numpy scikit-learn streamlit plotly
2. Run ML Prediction
bash
Copy
Edit
python ml_prediction.py
# or enhanced version
python ml_enhancement.py
3. Run Dashboard
bash
Copy
Edit
streamlit run dashboard.py
4. Optional: Set up Email Alerts
Generate Gmail App Password.

Replace placeholder in ml_enhancement.py.

📌 Highlights
✅ Time-series aware ML pipeline

✅ Custom trading strategy logic

✅ Performance metrics & visualization

✅ Portfolio simulation with real-world constraints

✅ Gmail email integration for alerts

✅ Fully modular and production-friendly







👨‍💻 Author
Abhay Kush
Data Analyst | Python Enthusiast | ML Explorer
📧 [abhaykush584@gmail.com]
🔗 [www.linkedin.com/in/abhay-kush-440696259]
