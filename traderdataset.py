import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

sns.set(style="whitegrid")


# Loading and Cleaning Sentiment Dataset

sentiment_df = pd.read_csv("fear_greed_index.csv")
sentiment_df.columns = sentiment_df.columns.str.strip()
sentiment_df.rename(columns={'date': 'Date', 'classification': 'Classification'}, inplace=True)
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

# Loading and Cleaning Trader Dataset

trader_df = pd.read_csv("historical_data.csv")
trader_df.columns = trader_df.columns.str.strip()
trader_df.rename(columns={'Timestamp': 'timestamp_ms', 'Closed PnL': 'profit'}, inplace=True)

# Convert epoch milliseconds to datetime
trader_df['execution_time'] = pd.to_datetime(trader_df['timestamp_ms'], unit='ms', errors='coerce')
trader_df['Date'] = pd.to_datetime(trader_df['execution_time'].dt.date)

# Optional debug
print("\nParsed execution times:\n", trader_df['execution_time'].head(3))
print("Parsed trader dates:\n", trader_df['Date'].unique()[:5])

#  Pre-Merge Diagnostics

print("\nSentiment unique dates:", sentiment_df['Date'].dt.date.unique()[:5])
print("Trader unique dates:", trader_df['Date'].dt.date.unique()[:5])
common_dates = set(sentiment_df['Date']).intersection(set(trader_df['Date']))
print("\n✅ Common Dates Between Sentiment and Trader Data:", len(common_dates))

# Merging Both Datasets

merged_df = pd.merge(trader_df, sentiment_df, on='Date', how='inner')
print("✅ Merged shape:", merged_df.shape)
print("Sample:\n", merged_df[['Date', 'profit', 'Classification']].head())

# Summary Statistics

performance_by_sentiment = merged_df.groupby('Classification')['profit'].agg(['mean', 'median', 'std', 'count']).reset_index()
print("\n📊 Profit summary:\n", performance_by_sentiment)

# Saving file
performance_by_sentiment.to_csv("sentiment_vs_profit_summary.csv", index=False)

# Boxplot (plotting+saving file)

plt.figure(figsize=(8, 5))
sns.boxplot(x='Classification', y='profit', data=merged_df)
plt.title("Profit Distribution by Market Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("profit_by_sentiment_boxplot.png")
plt.close()

# Lineplot (plotting+saving)

daily_avg = merged_df.groupby(['Date', 'Classification'])['profit'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_avg, x='Date', y='profit', hue='Classification', marker='o')
plt.title("Daily Average Profit vs Market Sentiment")
plt.xlabel("Date")
plt.ylabel("Avg Profit")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_avg_profit_lineplot.png")
plt.close()

# C R

merged_df['sentiment_score'] = merged_df['Classification'].map({'Fear': 0, 'Greed': 1})
correlation = merged_df['sentiment_score'].corr(merged_df['profit'])
print(f"\n📈 Correlation between Sentiment and Profit: {correlation:.4f}")