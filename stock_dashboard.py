
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import smtplib
from email.mime.text import MIMEText
import io

# Page config (must be first Streamlit command)
st.set_page_config(page_title="AAPL Trading Dashboard", layout="wide")

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Custom CSS for light and dark themes, including animations
light_css = """
<style>
body, .stApp {
    background-color: #ffffff;
    color: #000000;
    transition: background-color 0.3s ease;
}
.stSidebar {
    background-color: #f0f2f6;
}
.stPlotlyChart, .stDataFrame {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    animation: fadeIn 0.5s ease-in;
}
.theme-toggle {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
}
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
    animation: slideIn 0.5s ease-in;
}
.card {
    background-color: #f0f2f6;
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    color: #000000;
    animation: fadeIn 0.5s ease-in;
    transition: transform 0.2s ease;
}
.card:hover {
    transform: scale(1.05);
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
</style>
"""

dark_css = """
<style>
body, .stApp {
    background-color: #1e1e1e;
    color: #ffffff;
    transition: background-color 0.3s ease;
}
.stSidebar {
    background-color: #2c2c2c;
}
.stPlotlyChart, .stDataFrame {
    background-color: #2c2c2c;
    border: 1px solid #444444;
    animation: fadeIn 0.5s ease-in;
}
.theme-toggle {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    animation: slideIn 0.5s ease-in;
}
.card {
    background-color: #2c2c2c;
    border: 1px solid #444444;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    color: #ffffff;
    animation: fadeIn 0.5s ease-in;
    transition: transform 0.2s ease;
}
.card:hover {
    transform: scale(1.05);
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
</style>
"""

# Apply theme based on session state
if st.session_state.theme == 'dark':
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

# Theme toggle button in top-right corner
st.markdown(
    '<div class="theme-toggle">',
    unsafe_allow_html=True
)
if st.button(f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"):
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Title and header
st.title("Stock Trading Recommendation System: AAPL Dashboard")
st.markdown("Visualizing ML-Predicted Prices and Trading Performance")

# Load data
@st.cache_data
def load_data():
    try:
        predictions_df = pd.read_csv('predictions_enhanced_aapl.csv')
        trades_df = pd.read_csv('trades_aapl.csv')
        portfolio_df = pd.read_csv('portfolio_aapl.csv')
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        trades_df['Buy_Date'] = pd.to_datetime(trades_df['Buy_Date'])
        trades_df['Sell_Date'] = pd.to_datetime(trades_df['Sell_Date'])
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        return predictions_df, trades_df, portfolio_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure CSVs are in the project folder.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

predictions_df, trades_df, portfolio_df = load_data()
if predictions_df is None:
    st.stop()

# Check trades_df columns
required_trade_cols = ['Buy_Date', 'Buy_Price', 'Sell_Date', 'Sell_Price', 'Profit', 'Time_Frame']
if not all(col in trades_df.columns for col in required_trade_cols):
    st.error("Error: trades_aapl.csv missing required columns: " + ", ".join(required_trade_cols))
    st.stop()

# Sidebar for interactivity
st.sidebar.header("Navigation")
nav_option = st.sidebar.radio(
    "Best Fit to See Graphs",
    ["Full Dashboard", "Graphs Only"]
)

st.sidebar.header("Filter Options")
view = st.sidebar.selectbox("Select View", ["Hourly", "Daily"])
date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=predictions_df['Date'].min().to_pydatetime(),
    max_value=predictions_df['Date'].max().to_pydatetime(),
    value=(predictions_df['Date'].min().to_pydatetime(), predictions_df['Date'].max().to_pydatetime()),
    format="YYYY-MM-DD"
)

# Email alerts configuration
st.sidebar.header("Email Alerts")
email_enabled = st.sidebar.checkbox("Enable Email Alerts")
recipient_email = st.sidebar.text_input("Recipient Email", "")
smtp_password = st.sidebar.text_input("Gmail App Password", type="password")

# Download buttons
st.sidebar.header("Export Data")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if not predictions_df.empty:
    st.sidebar.download_button(
        label="Download Predictions",
        data=convert_df_to_csv(predictions_df),
        file_name="predictions_enhanced_aapl.csv",
        mime="text/csv"
    )
if not trades_df.empty:
    st.sidebar.download_button(
        label="Download Trades",
        data=convert_df_to_csv(trades_df),
        file_name="trades_aapl.csv",
        mime="text/csv"
    )
if not portfolio_df.empty:
    st.sidebar.download_button(
        label="Download Portfolio",
        data=convert_df_to_csv(portfolio_df),
        file_name="portfolio_aapl.csv",
        mime="text/csv"
    )

# Stocks Tracker in sidebar (cards and gauge)
st.sidebar.header("Stocks Tracker")
if not trades_df.empty:
    total_trades = len(trades_df)
    avg_profit = trades_df['Profit'].mean()
    total_profit = trades_df['Profit'].sum()
    win_rate = (trades_df['Profit'] > 0).mean() * 100
    filtered_buy_signals = len(trades_df[
        (trades_df['Buy_Date'] >= pd.to_datetime(date_range[0])) &
        (trades_df['Buy_Date'] <= pd.to_datetime(date_range[1]))
    ])
    filtered_sell_signals = len(trades_df[
        (trades_df['Sell_Date'] >= pd.to_datetime(date_range[0])) &
        (trades_df['Sell_Date'] <= pd.to_datetime(date_range[1]))
    ])
    
    # Cards
    st.sidebar.markdown(f'<div class="card">Total Trades: {total_trades}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="card">Average Profit: ${avg_profit:.2f}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="card">Total Profit: ${total_profit:.2f}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="card">Buy Signals: {filtered_buy_signals}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="card">Sell Signals: {filtered_sell_signals}</div>', unsafe_allow_html=True)
    
    # Gauge for Win Rate
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        title={'text': "Win Rate (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#000000" if st.session_state.theme == 'light' else "#ffffff"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "#ffffff" if st.session_state.theme == 'light' else "#2c2c2c",
            'bordercolor': "#e6e6e6" if st.session_state.theme == 'light' else "#444444"
        }
    ))
    fig_gauge.update_layout(
        font=dict(size=12, color='#000000' if st.session_state.theme == 'light' else '#ffffff'),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c'
    )
    st.sidebar.plotly_chart(fig_gauge, use_container_width=True)
else:
    st.sidebar.warning("No trade data available.")

# Filter data by date range
filtered_predictions = predictions_df[
    (predictions_df['Date'] >= pd.to_datetime(date_range[0])) &
    (predictions_df['Date'] <= pd.to_datetime(date_range[1]))
]
filtered_trades = trades_df[
    (trades_df['Buy_Date'] >= pd.to_datetime(date_range[0])) &
    (trades_df['Sell_Date'] <= pd.to_datetime(date_range[1]))
]
filtered_portfolio = portfolio_df[
    (portfolio_df['Date'] >= pd.to_datetime(date_range[0])) &
    (portfolio_df['Date'] <= pd.to_datetime(date_range[1]))
]

# Email alerts for new signals
def send_email(subject, body, recipient, password):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'abhaykush050804@gmail.com'  # Replace with your Gmail
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login('akks1925@gmail.com', password)
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

if email_enabled and recipient_email and smtp_password and not filtered_trades.empty:
    recent_trades = filtered_trades[
        filtered_trades['Buy_Date'] >= pd.to_datetime(datetime.now() - timedelta(hours=1))
    ]
    for _, trade in recent_trades.iterrows():
        signal_type = "Buy" if trade['Buy_Price'] else "Sell"
        subject = f"{signal_type} Signal for AAPL"
        body = f"{signal_type} signal detected on {trade['Buy_Date'] if signal_type == 'Buy' else trade['Sell_Date']}.\nPrice: ${trade['Buy_Price'] if signal_type == 'Buy' else trade['Sell_Price']:.2f}"
        if send_email(subject, body, recipient_email, smtp_password):
            st.sidebar.success(f"Email sent for {signal_type} signal!")

# Aggregate for daily view
if view == "Daily" and not filtered_trades.empty:
    daily_trades = []
    grouped = filtered_trades.groupby(filtered_trades['Buy_Date'].dt.date)
    for date, group in grouped:
        sell_date = group['Sell_Date'].min() if not group['Sell_Date'].isna().all() else pd.to_datetime(date)
        daily_trade = {
            'Buy_Date': pd.to_datetime(date),
            'Buy_Price': group['Buy_Price'].mean(),
            'Sell_Price': group['Sell_Price'].mean(),
            'Profit': group['Profit'].sum(),
            'Time_Frame': group['Time_Frame'].iloc[0],
            'Sell_Date': sell_date
        }
        daily_trades.append(daily_trade)
    filtered_trades = pd.DataFrame(daily_trades)
    if filtered_trades['Sell_Date'].isna().any():
        st.warning("Some Sell_Date values are missing in daily view. Using Buy_Date as fallback.")
        filtered_trades['Sell_Date'] = filtered_trades['Sell_Date'].fillna(filtered_trades['Buy_Date'])
    
    filtered_predictions = filtered_predictions.resample('D', on='Date').mean().reset_index()
    filtered_predictions['Date'] = pd.to_datetime(filtered_predictions['Date'])
    filtered_portfolio = filtered_portfolio.resample('D', on='Date').mean().reset_index()
    filtered_portfolio['Date'] = pd.to_datetime(filtered_portfolio['Date'])

# Define theme-aware font color
font_color = '#000000' if st.session_state.theme == 'light' else '#ffffff'

# Price trend plot (enhanced line plot)
fig_price = px.line(filtered_predictions, x='Date', y=['Actual_Close', 'Predicted_Close'],
                    title="AAPL Price Trends with Buy/Sell Signals",
                    labels={'value': 'Price ($)', 'variable': 'Price Type'})
fig_price.update_traces(line=dict(width=3))
fig_price.update_traces(selector=dict(name='Actual_Close'), line=dict(color='#1f77b4'))
fig_price.update_traces(selector=dict(name='Predicted_Close'), line=dict(color='#ff7f0e'))
if not filtered_trades.empty:
    buy_signals = go.Scatter(
        x=filtered_trades['Buy_Date'],
        y=filtered_trades['Buy_Price'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', size=12, color='green')
    )
    fig_price.add_trace(buy_signals)
    if 'Sell_Date' in filtered_trades.columns and not filtered_trades['Sell_Date'].isna().all():
        sell_signals = go.Scatter(
            x=filtered_trades['Sell_Date'],
            y=filtered_trades['Sell_Price'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=12, color='red')
        )
        fig_price.add_trace(sell_signals)
    else:
        st.warning("Sell signals not plotted due to missing or invalid Sell_Date.")
fig_price.update_layout(
    hovermode='x unified',
    plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
    paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
    font=dict(size=12, color=font_color),
    xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
    yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
    showlegend=True,
    transition=dict(duration=500, easing='cubic-in-out')
)
st.plotly_chart(fig_price, use_container_width=True)

# Day-wise Profit Margin Line Chart
if not filtered_trades.empty:
    profit_margin_df = filtered_trades.copy()
    profit_margin_df['Profit Margin (%)'] = ((profit_margin_df['Sell_Price'] - profit_margin_df['Buy_Price']) / profit_margin_df['Buy_Price']) * 100
    profit_margin_df['Date'] = profit_margin_df['Buy_Date'].dt.date
    profit_margin_daily = profit_margin_df.groupby('Date')['Profit Margin (%)'].sum().reset_index()
    profit_margin_daily['Date'] = pd.to_datetime(profit_margin_daily['Date'])
    if not profit_margin_daily.empty:
        fig_profit = px.line(
            profit_margin_daily,
            x='Date',
            y='Profit Margin (%)',
            title="Day-wise Profit Margin Trend",
            labels={'Profit Margin (%)': 'Profit Margin (%)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_profit.update_traces(
            line=dict(width=3),
            mode='lines+markers',
            marker=dict(size=8, symbol='circle'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Profit Margin: %{y:.2f}%'
        )
        fig_profit.update_layout(
            hovermode='x unified',
            plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
            paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
            font=dict(size=12, color=font_color),
            xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
            yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
            showlegend=False,
            transition=dict(duration=500, easing='cubic-in-out')
        )
        st.plotly_chart(fig_profit, use_container_width=True)
    else:
        st.warning("No profit margin data available for the selected date range.")
else:
    st.warning("No profit margin data available for the selected date range.")

# Portfolio value plot with markers
if not filtered_portfolio.empty:
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(
        go.Scatter(
            x=filtered_portfolio['Date'],
            y=filtered_portfolio['Portfolio_Value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='green', width=3),
            marker=dict(symbol='circle', size=8, color='green')
        )
    )
    fig_portfolio.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified',
        plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
        paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
        font=dict(size=12, color=font_color),
        xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
        yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
        showlegend=True,
        transition=dict(duration=500, easing='cubic-in-out')
    )
    st.plotly_chart(fig_portfolio, use_container_width=True)
else:
    st.warning("No portfolio data available for the selected date range.")

# Non-graph sections (shown only in Full Dashboard mode)
if nav_option == "Full Dashboard":
    # Insights and Backtest Statistics
    st.subheader("Insights and Backtest Statistics")
    if not filtered_trades.empty and not filtered_portfolio.empty:
        total_profit = filtered_trades['Profit'].sum()
        win_rate = (trades_df['Profit'] > 0).mean() * 100
        initial_cash = 10000
        final_value = filtered_portfolio['Portfolio_Value'].iloc[-1]
        growth = ((final_value - initial_cash) / initial_cash) * 100
        days = (filtered_portfolio['Date'].max() - filtered_portfolio['Date'].min()).days
        annualized_return = ((final_value / initial_cash) ** (365.0 / days) - 1) * 100 if days > 0 else 0
        returns = filtered_trades['Profit'] / filtered_trades['Buy_Price']
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6) if len(returns) > 1 else 0
        drawdown = ((filtered_portfolio['Portfolio_Value'].cummax() - filtered_portfolio['Portfolio_Value']) / filtered_portfolio['Portfolio_Value'].cummax() * 100).max()
        win_loss_ratio = (trades_df['Profit'] > 0).sum() / (trades_df['Profit'] < 0).sum() if (trades_df['Profit'] < 0).sum() > 0 else float('inf')
        avg_trade_duration = (trades_df['Sell_Date'] - trades_df['Buy_Date']).mean().total_seconds() / 3600 if not trades_df.empty else 0
        
        insights = f"""
        **Insights**:
        - **Profitability**: Total profit of ${total_profit:.2f} with a {win_rate:.1f}% win rate indicates strong model performance.
        - **Growth**: Portfolio grew by {growth:.2f}%, suggesting effective trading strategy.
        - **Risk**: Max drawdown of {drawdown:.2f}% highlights potential risk exposure; monitor closely.
        - **Efficiency**: Sharpe ratio of {sharpe_ratio:.2f} reflects solid risk-adjusted returns.
        """
        st.markdown(insights)
        
        backtest_df = pd.DataFrame({
            'Metric': ['Annualized Return (%)', 'Max Drawdown (%)', 'Win/Loss Ratio', 'Average Trade Duration (Hours)'],
            'Value': [f"{annualized_return:.2f}", f"{drawdown:.2f}", f"{win_loss_ratio:.2f}", f"{avg_trade_duration:.2f}"]
        })
        st.subheader("Backtest Statistics")
        st.dataframe(backtest_df, use_container_width=True)
    else:
        st.warning("No data available for insights or backtest statistics.")

    # Buy/Sell Records Table
    if not filtered_trades.empty:
        records_df = filtered_trades[['Buy_Date', 'Buy_Price', 'Sell_Date', 'Sell_Price', 'Profit']].copy()
        records_df['Profit Margin (%)'] = ((records_df['Sell_Price'] - records_df['Buy_Price']) / records_df['Buy_Price']) * 100
        records_df['Buy_Date'] = records_df['Buy_Date'].dt.strftime('%Y-%m-%d %H:%M')
        records_df['Sell_Date'] = records_df['Sell_Date'].dt.strftime('%Y-%m-%d %H:%M')
        records_df = records_df.rename(columns={
            'Buy_Date': 'Buy Date (Hour)',
            'Buy_Price': 'Buy Price ($)',
            'Sell_Date': 'Sell Date (Hour)',
            'Sell_Price': 'Sell Price ($)',
            'Profit': 'Profit ($)',
            'Profit Margin (%)': 'Profit Margin (%)'
        })
        st.subheader("Buy/Sell Records")
        st.dataframe(records_df, use_container_width=True)
    else:
        st.warning("No trade records available for the selected date range.")

    # Performance metrics
    if not filtered_trades.empty and not filtered_portfolio.empty:
        total_profit = filtered_trades['Profit'].sum()
        win_rate = (trades_df['Profit'] > 0).mean() * 100
        initial_cash = 10000
        final_value = filtered_portfolio['Portfolio_Value'].iloc[-1]
        growth = ((final_value - initial_cash) / initial_cash) * 100
        cash_ratio = filtered_portfolio['Cash'].iloc[-1] / final_value
        diversification = "Balanced" if 0.4 <= cash_ratio <= 0.6 else "Unbalanced"
        returns = filtered_trades['Profit'] / filtered_trades['Buy_Price']
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6) if len(returns) > 1 else 0
        performance = "Strong" if growth > 2 else "Moderate" if growth > 0 else "Loss"
        
        metrics_df = pd.DataFrame({
            'Metric': ['Total Profit ($)', 'Win Rate (%)', 'Growth (%)', 'Diversification', 'Sharpe Ratio', 'Performance'],
            'Value': [f"{total_profit:.2f}", f"{win_rate:.1f}", f"{growth:.2f}", diversification, f"{sharpe_ratio:.2f}", performance]
        })
        st.subheader("Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("No trades or portfolio data available for the selected date range.")

    # Visual Representation section
    st.subheader("Visual Representation")
    
    # Stock Held by Users
    with st.expander("Stock Held by Users"):
        if not filtered_trades.empty:
            shares_df = pd.DataFrame()
            buy_shares = filtered_trades[['Buy_Date']].copy()
            buy_shares['Shares'] = 1
            buy_shares['Type'] = 'Buy'
            sell_shares = filtered_trades[['Sell_Date']].copy()
            sell_shares['Shares'] = -1
            sell_shares['Type'] = 'Sell'
            sell_shares = sell_shares.rename(columns={'Sell_Date': 'Buy_Date'})
            shares_df = pd.concat([buy_shares, sell_shares], ignore_index=True)
            shares_df['Date'] = pd.to_datetime(shares_df['Buy_Date']).dt.date
            shares_df = shares_df.groupby('Date')['Shares'].sum().reset_index()
            shares_df['Date'] = pd.to_datetime(shares_df['Date'])
            shares_df['Cumulative Shares'] = shares_df['Shares'].cumsum()
            
            fig_shares = px.bar(
                shares_df,
                x='Date',
                y='Cumulative Shares',
                title="Stock Held by Users (Cumulative Shares)",
                labels={'Cumulative Shares': 'Shares Held'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_shares.update_traces(
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Shares Held: %{y}'
            )
            fig_shares.update_layout(
                hovermode='x unified',
                plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                font=dict(size=12, color=font_color),
                xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                showlegend=False,
                transition=dict(duration=500, easing='cubic-in-out')
            )
            st.plotly_chart(fig_shares, use_container_width=True)
        else:
            st.warning("No trade data available for stock holdings.")
    
    # Profit Share (Hourly/Monthly)
    with st.expander("Profit Share (Hourly/Monthly)"):
        if not filtered_trades.empty:
            time_frame = st.selectbox("Select Time Frame", ["Hourly", "Monthly"], key="profit_share_time")
            profit_df = filtered_trades[['Buy_Date', 'Profit']].copy()
            profit_df['Date'] = pd.to_datetime(profit_df['Buy_Date'])
            if time_frame == "Hourly":
                profit_df = profit_df.resample('H', on='Date')['Profit'].sum().reset_index()
            else:
                profit_df = profit_df.resample('M', on='Date')['Profit'].sum().reset_index()
            
            fig_profit_share = px.area(
                profit_df,
                x='Date',
                y='Profit',
                title=f"Profit Share ({time_frame})",
                labels={'Profit': 'Profit ($)'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig_profit_share.update_traces(
                hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Profit: $%{y:.2f}' if time_frame == "Hourly" else 'Date: %{x|%Y-%m}<br>Profit: $%{y:.2f}'
            )
            fig_profit_share.update_layout(
                hovermode='x unified',
                plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                font=dict(size=12, color=font_color),
                xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                showlegend=False,
                transition=dict(duration=500, easing='cubic-in-out')
            )
            st.plotly_chart(fig_profit_share, use_container_width=True)
        else:
            st.warning("No trade data available for profit share.")
    
    # Stock Value Rises (Hourly/Daily)
    with st.expander("Stock Value Rises (Hourly/Daily)"):
        if not filtered_predictions.empty:
            time_frame = st.selectbox("Select Time Frame", ["Hourly", "Daily"], key="value_rises_time")
            value_df = filtered_predictions[['Date', 'Actual_Close']].copy()
            value_df['Price Change'] = value_df['Actual_Close'].diff()
            if time_frame == "Daily":
                value_df = value_df.resample('D', on='Date').mean().reset_index()
                value_df['Price Change'] = value_df['Actual_Close'].diff()
            
            fig_value_rises = px.line(
                value_df,
                x='Date',
                y='Price Change',
                title=f"Stock Value Rises ({time_frame})",
                labels={'Price Change': 'Price Change ($)'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_value_rises.update_traces(
                line=dict(width=3),
                mode='lines+markers',
                marker=dict(size=8, symbol='circle'),
                hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Price Change: $%{y:.2f}' if time_frame == "Hourly" else 'Date: %{x|%Y-%m-%d}<br>Price Change: $%{y:.2f}'
            )
            fig_value_rises.update_layout(
                hovermode='x unified',
                plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                font=dict(size=12, color=font_color),
                xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                showlegend=False,
                transition=dict(duration=500, easing='cubic-in-out')
            )
            st.plotly_chart(fig_value_rises, use_container_width=True)
        else:
            st.warning("No price data available for stock value rises.")
    
    # Portfolio Drawdown
    with st.expander("Portfolio Drawdown"):
        if not filtered_portfolio.empty:
            drawdown_df = filtered_portfolio[['Date', 'Portfolio_Value']].copy()
            drawdown_df['Peak'] = drawdown_df['Portfolio_Value'].cummax()
            drawdown_df['Drawdown (%)'] = (drawdown_df['Peak'] - drawdown_df['Portfolio_Value']) / drawdown_df['Peak'] * 100
            
            fig_drawdown = px.line(
                drawdown_df,
                x='Date',
                y='Drawdown (%)',
                title="Portfolio Drawdown Over Time",
                labels={'Drawdown (%)': 'Drawdown (%)'},
                color_discrete_sequence=['#d62728']
            )
            fig_drawdown.update_traces(
                line=dict(width=3),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%'
            )
            fig_drawdown.update_layout(
                hovermode='x unified',
                plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                font=dict(size=12, color=font_color),
                xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                showlegend=False,
                transition=dict(duration=500, easing='cubic-in-out')
            )
            st.plotly_chart(fig_drawdown, use_container_width=True)
        else:
            st.warning("No portfolio data available for drawdown.")
    
    # Sharpe Ratio Trend
    with st.expander("Sharpe Ratio Trend"):
        if not filtered_trades.empty:
            sharpe_df = filtered_trades[['Buy_Date', 'Profit', 'Buy_Price']].copy()
            sharpe_df['Date'] = pd.to_datetime(sharpe_df['Buy_Date']).dt.date
            sharpe_df['Returns'] = sharpe_df['Profit'] / sharpe_df['Buy_Price']
            sharpe_daily = sharpe_df.groupby('Date')['Returns'].mean().reset_index()
            sharpe_daily['Date'] = pd.to_datetime(sharpe_daily['Date'])
            sharpe_daily['Rolling_Mean'] = sharpe_daily['Returns'].rolling(window=30, min_periods=1).mean()
            sharpe_daily['Rolling_Std'] = sharpe_daily['Returns'].rolling(window=30, min_periods=1).std()
            sharpe_daily['Sharpe_Ratio'] = (sharpe_daily['Rolling_Mean'] / sharpe_daily['Rolling_Std'] * np.sqrt(252 * 6)).fillna(0)
            
            fig_sharpe = px.line(
                sharpe_daily,
                x='Date',
                y='Sharpe_Ratio',
                title="Rolling Sharpe Ratio (30-Day Window)",
                labels={'Sharpe_Ratio': 'Sharpe Ratio'},
                color_discrete_sequence=['#9467bd']
            )
            fig_sharpe.update_traces(
                line=dict(width=3),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Sharpe Ratio: %{y:.2f}'
            )
            fig_sharpe.update_layout(
                hovermode='x unified',
                plot_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                paper_bgcolor='#ffffff' if st.session_state.theme == 'light' else '#2c2c2c',
                font=dict(size=12, color=font_color),
                xaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                yaxis=dict(title_font=dict(color=font_color), tickfont=dict(color=font_color)),
                showlegend=False,
                transition=dict(duration=500, easing='cubic-in-out')
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        else:
            st.warning("No trade data available for Sharpe ratio.")
