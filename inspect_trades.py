import pandas as pd
trades_df = pd.read_csv('trades_aapl.csv')
print("Columns:", trades_df.columns.tolist())
print("Head:\n", trades_df.head())
print("Null Sell_Date count:", trades_df['Sell_Date'].isna().sum())
print("Sell_Date dtype:", trades_df['Sell_Date'].dtype)
print("Total rows:", len(trades_df))

