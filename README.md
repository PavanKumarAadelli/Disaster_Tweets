# 🌩️ Disaster Tweets Analysis & Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project analyzes a dataset of tweets related to natural disasters to understand public engagement trends and build a Machine Learning model capable of automatically classifying the type of disaster based on the tweet's text.

## 📝 Project Overview

Natural disasters are increasingly discussed on social media platforms. This project leverages Natural Language Processing (NLP) to analyze thousands of tweets, identifying key patterns in language and engagement across different disaster types (Droughts, Wildfires, Floods, Earthquakes, etc.).

**Key Features:**
*   **Data Cleaning:** Handling missing values, timestamp conversion, and removing non-disaster noise (e.g., sports tweets).
*   **Exploratory Data Analysis (EDA):** Visualizing tweet frequency and engagement (Likes/Retweets).
*   **NLP & Text Mining:** Generating Word Clouds and extracting hashtags.
*   **Machine Learning:** A Multinomial Naive Bayes classifier trained to predict disaster categories with an accuracy of **[Insert Accuracy, e.g., 85%]**.

## 🚀 Key Findings

*   **Most Frequent Topic:** Tweets about **Drought** were the most prevalent in the dataset.
*   **Highest Engagement:** **Earthquake** alerts received the highest average number of retweets, likely due to their urgent nature.
*   **Text Patterns:** Words like "rain," "water," and "fire" were dominant across different categories, requiring careful feature engineering to distinguish contexts.
*   **Noise Filtering:** Successfully filtered out sports-related tweets (e.g., "Carolina Hurricanes" hockey team) that were incorrectly labeled as natural disasters.

## 🛠️ Tech Stack

*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **NLP:** WordCloud, Regular Expressions (re)
*   **Machine Learning:** Scikit-Learn (TF-IDF Vectorizer, Multinomial Naive Bayes)
*   **Model Persistence:** Joblib

## 📦 Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    bash
    git clone https://github.com/PavanKumarAadelli/Disaster_Tweets.git

2.  **Navigate to the project directory:**
    bash
    cd Disaster_Tweets

3.  **Install the required dependencies:**
    bash
    pip install -r requirements.txt

## 💻 Usage

### 1. Run the Analysis
Open the Jupyter Notebook (analysis.ipynb) or the Python script to view the data cleaning process, EDA visualizations, and model training steps.

### 2. Predict a New Disaster
You can use the saved model artifacts to make predictions on new, unseen text.

python
import joblib

# Load the saved model and vectorizer
model = joblib.load('models/disaster_classifier_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Input a new tweet
new_tweet = ["Heavy rains are causing the river to overflow in the city."]

# Preprocess and predict
tweet_vectorized = vectorizer.transform(new_tweet)
prediction = model.predict(tweet_vectorized)

print(f"Predicted Disaster Type: {prediction[0]}")

## 📊 Visualizations

### Disaster Tweet Distribution
[Bar Char Frequency.png]

### Word Cloud
[Word Cloud.png]

### Model Performance (Confusion Matrix)
[Confusion Matrix.png]

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).
