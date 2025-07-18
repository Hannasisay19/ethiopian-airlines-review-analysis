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

## Project Structure

```
ETHIOPIAN-AIRLINES-REVIEW-ANALYSIS/
├── datasets/           # Raw, cleaned, merged, and labeled review data
├── models/             # Trained ML and LSTM model files (.joblib, .pth)
├── project_code/       # Notebooks and scripts for scraping, cleaning, EDA, modeling, and dashboards
├── images/             # Visuals for model performance (confusion matrices, classification reports)
├── requirements.txt    # List of dependencies
├── README.md           # Project overview and instructions
├── .gitignore          # Files/directories to ignore in version control

```




# Steps to run the project 
Clone the repo and install the required libraries:
```bash
git clone https://github.com/Hannasisay19/ethiopian-airlines-review-analysis.git
cd ethiopian-airlines-review-analysis
pip install -r requirements.txt


# Dashboard Features 
Sentiment Pie Chart: Share of positive, neutral, and negative reviews.
Sentiment Bar Chart: Count of reviews per sentiment.
Average Ratings: Mean rating per category or selected category.
Sentiment Trend: Monthly sentiment counts over time.
Rating Trend: Monthly average ratings per category.
Top Cities:
   Top 10 departure cities with most positive/negative reviews.
   Top 10 arrival cities with most positive/negative reviews.
Reviews Table: Paginated table with filtered reviews.


# Models Used
Classical ML Models: SVM, Random Forest, Logistic Regression.
Deep Learning: LSTM
Performance evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.


# Key Insights
Most reviews are from recent years (2015–2025).
Negative sentiment is prevalent, but there is a notable proportion of positive feedback as well.


# License
This project is licensed under the MIT License.


# Acknowledgements
Thanks to:
airlinequality.com
TripAdvisor
Ethiopian Airlines passengers for the public feedback


# Author
Hanna Sisay Mengistu
Mekbib Lakew Gebreyohannes


Capstone Project — 2025

