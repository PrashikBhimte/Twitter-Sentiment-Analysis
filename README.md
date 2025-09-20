# Twitter Sentiment Analysis

This project aims to classify the sentiment of tweets into three categories: Positive, Negative, and Neutral. It explores two different approaches: a traditional machine learning model (TF-IDF with Logistic Regression) and a deep learning model (BERT).

## Project Structure

```
/home/prashikbhimte/projects/ML-Projects/2_Twitter_Sentiment_Analysis/
├───.python-version
├───README.md
├───requirements.txt
├───model_training/
│   ├───main.ipynb
│   └───sentimentdataset.csv
├───trained_models/
│   ├───bert_logistic_reg_model.joblib
│   ├───logistic_reg_model.joblib
│   └───tfidf_vectorizer.joblib
└───venv/
```

## Files

*   `model_training/main.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
*   `model_training/sentimentdataset.csv`: The dataset used for training and evaluation.
*   `trained_models/tfidf_vectorizer.joblib`: The saved TF-IDF vectorizer.
*   `trained_models/logistic_reg_model.joblib`: The saved Logistic Regression model.
*   `trained_models/bert_logistic_reg_model.joblib`: The saved BERT model.
*   `requirements.txt`: A list of Python packages required to run the project.
*   `README.md`: This file.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd 2_Twitter_Sentiment_Analysis
    ```
3.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4.  Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
5.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the trained models for sentiment prediction, you can load the saved models and the TF-IDF vectorizer in a Python script. Here is an example of how to predict the sentiment of a new tweet using the Logistic Regression model:

```python
import joblib

# Load the trained model and vectorizer
model = joblib.load('trained_models/logistic_reg_model.joblib')
vectorizer = joblib.load('trained_models/tfidf_vectorizer.joblib')

# Function to clean the text (should be the same as in the notebook)
def clean_text(text):
    # ... (add the text cleaning function here)
    return cleaned_text

# New tweet to classify
new_tweet = "This is a great movie!"

# Clean the tweet
cleaned_tweet = clean_text(new_tweet)

# Vectorize the tweet
vectorized_tweet = vectorizer.transform([cleaned_tweet])

# Predict the sentiment
prediction = model.predict(vectorized_tweet)

print(f"The sentiment of the tweet is: {prediction[0]}")
```

## Models

Two models were trained for this project:

1.  **Logistic Regression with TF-IDF**: A traditional machine learning model that uses TF-IDF to represent the text data. The model was trained on a balanced dataset to handle class imbalance.

2.  **BERT**: A pre-trained transformer-based model that was fine-tuned for the sentiment analysis task. This model generally provides higher accuracy than the traditional approach.