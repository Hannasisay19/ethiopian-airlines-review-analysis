import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Load the data
df = pd.read_csv(r'datasets\labeled_data\ethiopian_airlines_overall_and_topic_sentiment.csv')
# Pie chart of overall sentiment
df['overall_sentiment'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#99ff99'])
plt.title('Overall Sentiment Proportion')
plt.ylabel('')
plt.show()

# --- Most Negative Topics per Review ---
# If 'negative_topics' not in df, define it here:
topic_cols = [col for col in df.columns if col.endswith('_sentiment') and col != 'overall_sentiment']
if 'negative_topics' not in df.columns:
    def most_negative_topic(row):
        return [col for col in topic_cols if row[col] == 'Negative'] or ['None']
    df['negative_topics'] = df.apply(most_negative_topic, axis=1)

negative_flat = [item for sublist in df['negative_topics'] for item in sublist]
neg_topic_counts = Counter(negative_flat)
if 'None' in neg_topic_counts:
    del neg_topic_counts['None']

neg_topic_df = pd.DataFrame(neg_topic_counts.items(), columns=['Topic', 'Negative_Count'])
neg_topic_df['Percentage'] = 100 * neg_topic_df['Negative_Count'] / len(df)
neg_topic_df = neg_topic_df.sort_values('Negative_Count', ascending=False).reset_index(drop=True)

print("\nTable: Topics most likely to be negative per review")
print(neg_topic_df)

plt.figure(figsize=(8,4))
sns.barplot(data=neg_topic_df, x='Negative_Count', y='Topic', palette='Reds_r')
plt.title('Top Topics Most Likely to be Negative')
plt.xlabel('Number of Reviews Marked Negative')
plt.ylabel('Topic')
plt.show()

# --- Standardize Text Case ---
for col in topic_cols + ['overall_sentiment']:
    df[col] = df[col].str.strip().str.capitalize()

# --- Probability Heatmaps of Overall Sentiment given Topic Sentiment ---
heatmap_data = []
for topic in topic_cols:
    for topic_val in ['Negative', 'Neutral', 'Positive']:
        mask = df[topic] == topic_val
        count = mask.sum()
        if count == 0:
            pct_neg = pct_neu = pct_pos = np.nan
        else:
            pct_neg = (df.loc[mask, 'overall_sentiment'] == 'Negative').mean()
            pct_neu = (df.loc[mask, 'overall_sentiment'] == 'Neutral').mean()
            pct_pos = (df.loc[mask, 'overall_sentiment'] == 'Positive').mean()
        heatmap_data.append({
            'Topic': topic,
            'Topic Sentiment': topic_val,
            'Pct Overall Negative': pct_neg,
            'Pct Overall Neutral': pct_neu,
            'Pct Overall Positive': pct_pos
        })

hm_df = pd.DataFrame(heatmap_data)

short_labels = {
    'cabin_crew_sentiment': 'Cabin Crew',
    'flight_delay_sentiment': 'Flight Delay',
    'luggage_handling_sentiment': 'Luggage',
    'food_service_sentiment': 'Food',
    'seat_comfort_sentiment': 'Seat Comfort',
    'restroom_quality_sentiment': 'Restroom',
    'airport_check_sentiment': 'Airport Check',
    'customer_service_sentiment': 'Customer Service',
    'value_for_money_sentiment': 'Value',
    'inflight_entertainment_sentiment': 'Entertainment'
}
hm_df['Topic_Short'] = hm_df['Topic'].map(short_labels)

# Plot heatmap for "Pct Overall Negative"
neg_pivot = hm_df.pivot(index='Topic_Short', columns='Topic Sentiment', values='Pct Overall Negative')
plt.figure(figsize=(8,6))
sns.heatmap(neg_pivot, annot=True, fmt=".1%", cmap="Reds")
plt.title("Probability of Overall Sentiment being Negative, by Topic Sentiment")
plt.xlabel("Topic Sentiment")
plt.ylabel("Topic")
plt.tight_layout()
plt.show()

# Plot heatmap for "Pct Overall Positive"
pos_pivot = hm_df.pivot(index='Topic_Short', columns='Topic Sentiment', values='Pct Overall Positive')
plt.figure(figsize=(8,6))
sns.heatmap(pos_pivot, annot=True, fmt=".1%", cmap="Greens")
plt.title("Probability of Overall Sentiment being Positive, by Topic Sentiment")
plt.xlabel("Topic Sentiment")
plt.ylabel("Topic")
plt.tight_layout()
plt.show()

# Plot heatmap for "Pct Overall Neutral"
neu_pivot = hm_df.pivot(index='Topic_Short', columns='Topic Sentiment', values='Pct Overall Neutral')
plt.figure(figsize=(8,6))
sns.heatmap(neu_pivot, annot=True, fmt=".1%", cmap="Blues")
plt.title("Probability of Overall Sentiment being Neutral, by Topic Sentiment")
plt.xlabel("Topic Sentiment")
plt.ylabel("Topic")
plt.tight_layout()
plt.show()
