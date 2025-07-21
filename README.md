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
├── images/             # Trained ML and LSTM model files (.joblib, .pth)
├── models/       # Notebooks and scripts for scraping, cleaning, EDA, modeling, and dashboards
├── project_code/             # Visuals for model performance (confusion matrices, classification reports)
├── .gitignore   # List of dependencies
├── README.md           # Project overview and instructions
├── requirements.txt        # Files/directories to ignore in version control

```




# Steps to run the project 
Clone the repo and install the required libraries:
```bash
git clone https://github.com/Hannasisay19/ethiopian-airlines-review-analysis.git
cd ethiopian-airlines-review-analysis
pip install -r requirements.txt


# Dashboard Features 
The project includes two interactive dashboards built using Plotly Dash:
-Overall Sentiment Dashboard: Presents overall customer sentiment (positive, neutral, negative), average ratings, and review trends over time.
-Topic-Based Sentiment Dashboard: Highlights sentiments for specific topics such as seat comfort, cabin service, food, and more.
Both dashboards feature visualizations like pie charts, bar charts, line plots, and filtered review tables to support insight discovery.


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

