# Ethiopian Airlines Reviews Dataset Collection and Analytics

Ethiopian Airlines Reviews Dataset Collection and Analytics is an end-to-end data project that automates the gathering, cleaning, analysis, and visualization of customer reviews. The goal is to uncover actionable insights about customer satisfaction and service improvement opportunities.

This project focuses on collecting, processing, and analyzing customer reviews related to Ethiopian Airlines, gathered from AirlineQuality (https://www.airlinequality.com/) and TripAdvisor (https://www.tripadvisor.com/). Using Selenium , we aim to gain insights into customer sentiments, satisfaction levels, and key areas needing improvement.


---

# Objectives
- Scrape and clean review data from both data sources.
- Perform sentiment analysis (Positive, Neutral, Negative).
- Build a machine learning & deep learning classification model.
- Visualize key findings through a custom dashboard.

---

# Data Sources
- AirlineQuality.com
- TripAdvisor.com

---

# Tools & Libraries Used
- Web Scraping: `Selenium`, 
- Data Analysis: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- NLP & ML: `scikit-learn`, `nltk`, `PyTorch`
- Deep Learning: `LSTM`
- Dashboard: `Plotly Dash`
- Other Tools: `emoji`, `tqdm`, `joblib`

---

# Project Structure
ETHIOPIAN-AIRLINES-REVIEW-ANALYSIS/
├── datasets/
│   ├── cleaned_data/
│   │   ├── cleaned_airlinequality_ethiopian_airlines_reviews.csv
│   │   ├── cleaned_tripadvisor_ethiopian_airlines_reviews.csv
│   ├── labeled_data/
│   │   ├── ethiopian_airlines_overall_and_category_sentiment.csv
│   │   ├── ethiopian_airlines_overall_sentiment_final.csv
│   ├── merged_data/
│   │   ├── merged_cleaned_ethiopian_airlines_reviews.csv
│   ├── raw_data/
│   │   ├── scraped_airlinequality_ethiopian_airlines_reviews.csv
│   │   ├── scraped_tripadvisor_ethiopian_airlines_reviews.csv
│   ├── scraped_data/
│       ├── scraped_airlinequality_ethiopian_airlines_review.csv
│       ├── scraped_tripadvisor_ethiopian_airlines_reviews.csv
│
├── images/
│   ├── Logistic_Regression/
│   ├── LSTM/
│   ├── Random_forest/
│   ├── SVM/
│
├── models/
│   ├── label_encoder.joblib
│   ├── logreg_model.joblib
│   ├── tfidf_vectorizer.joblib
│
├── project_code/
│   ├── cleaner_airlinequality_ethiopian_airlines_reviews.ipynb
│   ├── cleaner_tripadvisor_ethiopian_airlines_reviews.ipynb
│   ├── ethiopian_airlines_overall_and_category_sentiment.ipynb
│   ├── ethiopian_airlines_overall_sentiment_analysis_final.ipynb
│   ├── ethiopian_airlines_overall_sentiment_dashboard.ipynb
│   ├── logistic_regression_sentiment_.ipynb
│   ├── merge_cleaned_ethiopian_airlines_reviews.ipynb
│   ├── random_forest_sentiment.ipynb
│   ├── scraper_airlinequality_ethiopian_airlines_reviews.ipynb
│   ├── scraper_tripadvisor_ethiopian_airlines_reviews.ipynb
│
├── train_lstm_sentiment.py
├── train_svm_sentiment.py
├── requirements.txt
├── README.md
├── .gitignore


#  Steps to run the project 
Clone the repo and install the required libraries:
```bash
git clone https://github.com/Hannasisay19/ethiopian-airlines-review-analysis.git
cd ethiopian-airlines-review-analysis
pip install -r requirements.txt

Dashboard Features
Sentiment Pie Chart: Share of positive, neutral, and negative reviews.
Sentiment Bar Chart: Count of reviews per sentiment.
Average Ratings: Mean rating per category or selected category.
Sentiment Trend: Monthly sentiment counts over time.
Rating Trend: Monthly average ratings per category.
Top Cities:
   Top 10 departure cities with most positive/negative reviews.
   Top 10 arrival cities with most positive/negative reviews.
Reviews Table: Paginated table with filtered reviews.

Models Used
Classical ML Models: SVM, Random Forest, Logistic Regression.
Deep Learning: LSTM
Performance evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

Key Insights
Most reviews are from recent years (2015–2025).
Negative sentiment dominates but there is also Positive sentiment

License
This project is licensed under the MIT License.

Acknowledgements
Thanks to:
airlinequality.com
TripAdvisor
Ethiopian Airlines passengers for the public feedback

Author
Hanna Sisay Mengistu
Mekbib Lakew Gebreyohannes

Capstone Project — 2025

