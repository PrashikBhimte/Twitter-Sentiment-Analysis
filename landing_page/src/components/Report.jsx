import React from 'react';

const Report = () => {
  return (
    <section className="bg-gray-800 text-white py-20">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12">Project Report</h2>

        <div className="mb-12">
          <h3 className="text-3xl font-bold mb-4">1. Introduction</h3>
          <p className="text-lg leading-relaxed">
            This report details the process of building and evaluating a sentiment analysis model for
            Twitter data. The goal of this project is to classify tweets into three sentiment categories:
            Positive, Negative, and Neutral. Two different approaches were explored: a traditional
            machine learning model using TF-IDF with Logistic Regression, and a deep learning model
            using a pre-trained BERT model.
          </p>
        </div>

        <div className="mb-12">
          <h3 className="text-3xl font-bold mb-4">2. Data Loading and Preprocessing</h3>
          <h4 className="text-2xl font-semibold mb-2">2.1. Data Loading and Initial Exploration</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The dataset used is `sentimentdataset.csv`.</li>
            <li>The initial dataset contained 732 entries.</li>
            <li>The initial exploration of the dataset was done using pandas to understand its structure,
                identify missing values, and check for duplicates.</li>
            <li>Unnecessary columns like 'Unnamed: 0.1' and 'Unnamed: 0' were dropped.</li>
            <li>20 duplicate rows were found and removed, resulting in a dataset with 712 entries.</li>
            <li>Data types of 'Retweets' and 'Likes' columns were converted from float to integer, and the
                'Timestamp' column was converted to a datetime object.</li>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">2.2. Sentiment Label Consolidation</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The original dataset contained 279 unique sentiment labels.</li>
            <li>To simplify the classification task, these labels were mapped into three main categories:
                'Positive', 'Negative', and 'Neutral'.</li>
            <li>After mapping, the sentiment distribution was as follows:</li>
            <ul className="list-circle list-inside ml-8">
              <li>Positive: 430</li>
              <li>Negative: 160</li>
              <li>Neutral: 122</li>
            </ul>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">2.3. Exploratory Data Analysis (EDA)</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The distribution of tweet lengths was analyzed to see if there is a correlation between the
                length of a tweet and its sentiment.</li>
            <li>A box plot was created to visualize the distribution of tweet lengths for each sentiment
                category.</li>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">2.4. Text Cleaning</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>A text cleaning function was created to preprocess the tweet text. The cleaning process
                involved:</li>
            <ul className="list-circle list-inside ml-8">
              <li>Converting text to lowercase.</li>
              <li>Removing user mentions (`@username`).</li>
              <li>Removing URLs.</li>
              <li>Removing punctuation.</li>
              <li>Removing emojis.</li>
              <li>Tokenizing the text.</li>
              <li>Lemmatizing the tokens using NLTK's `WordNetLemmatizer`.</li>
              <li>Removing stopwords.</li>
            </ul>
          </ul>
        </div>

        <div className="mb-12">
          <h3 className="text-3xl font-bold mb-4">3. Model 1: TF-IDF with Logistic Regression</h3>
          <h4 className="text-2xl font-semibold mb-2">3.1. Feature Extraction</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The cleaned tweet text was converted into a numerical format using the TF-IDF (Term
                Frequency-Inverse Document Frequency) vectorization technique.</li>
            <li>The `TfidfVectorizer` from scikit-learn was used with a 'max_features` limit of 5000,
                resulting in a feature matrix of shape (712, 2237).</li>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">3.2. Model Training and Evaluation (Initial - Imbalanced Data)</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The data was split into training and testing sets (80/20 split) with stratification to maintain
                the sentiment distribution.</li>
            <li>A Logistic Regression model was trained on the initial imbalanced data.</li>
            <li>The initial evaluation showed an accuracy of 0.69.</li>
            <li>The classification report revealed poor performance on the 'Negative' and 'Neutral' classes:</li>
            <ul className="list-circle list-inside ml-8">
              <li>Negative: Precision=1.00, Recall=0.34, F1-score=0.51</li>
              <li>Neutral: Precision=1.00, Recall=0.08, F1-score=0.15</li>
              <li>Positive: Precision=0.66, Recall=1.00, F1-score=0.80</li>
            </ul>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">3.3. Handling Class Imbalance</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>To address the class imbalance issue, the 'class_weight='balanced'' parameter was used
                in the Logistic Regression model.</li>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">3.4. Model Training and Evaluation (After Resampling)</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The Logistic Regression model was retrained on the balanced dataset.</li>
            <li>The model's performance improved significantly after resampling. The accuracy increased
                to 0.84.</li>
            <li>The precision, recall, and F1-scores for the 'Negative' and 'Neutral' classes were much
                better:</li>
            <ul className="list-circle list-inside ml-8">
              <li>Negative: Precision=0.84, Recall=0.81, F1-score=0.83</li>
              <li>Neutral: Precision=0.85, Recall=0.44, F1-score=0.58</li>
              <li>Positive: Precision=0.84, Recall=0.97, F1-score=0.90</li>
            </ul>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">3.5. Model Saving</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The trained `TfidfVectorizer` and the final `LogisticRegression` model were saved to disk
                using 'joblib` for later use. The saved files are:</li>
            <ul className="list-circle list-inside ml-8">
              <li>`tfidf_vectorizer.joblib`</li>
              <li>`logistic_reg_model.joblib`</li>
            </ul>
          </ul>
        </div>

        <div className="mb-12">
          <h3 className="text-3xl font-bold mb-4">4. Model 2: BERT for Sentiment Analysis</h3>
          <h4 className="text-2xl font-semibold mb-2">4.1. Introduction to BERT</h4>
          <p className="text-lg leading-relaxed">
            A more advanced deep learning approach using a pre-trained BERT (Bidirectional Encoder
            Representations from Transformers) model was also implemented. BERT is a powerful
            language representation model that has shown state-of-the-art results on various NLP tasks.
          </p>
          <h4 className="text-2xl font-semibold mb-2 mt-6">4.2. Tokenization and Input Preparation</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The text data was tokenized using the `DistilBertTokenizer` from the `transformers` library.</li>
            <li>The tokenized text was then converted into input tensors (input IDs and attention masks)
                suitable for the BERT model. The resulting feature matrix had a shape of (712, 768).</li>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">4.3. Model Building and Training</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>A Logistic Regression model was trained on the BERT embeddings.</li>
            <li>The training data was resampled using `RandomOverSampler` to handle class imbalance.</li>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">4.4. Evaluation</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The fine-tuned BERT model was evaluated on the test set. The results showed a high level
                of accuracy: 0.83.</li>
            <li>The classification report for the BERT-based model:</li>
            <ul className="list-circle list-inside ml-8">
              <li>Negative: Precision=0.79, Recall=0.84, F1-score=0.82</li>
              <li>Neutral: Precision=0.55, Recall=0.48, F1-score=0.51</li>
              <li>Positive: Precision=0.92, Recall=0.93, F1-score=0.92</li>
            </ul>
          </ul>
          <h4 className="text-2xl font-semibold mb-2 mt-6">4.5. Model Saving</h4>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The trained BERT-based model was also saved. The file is:</li>
            <ul className="list-circle list-inside ml-8">
              <li>`bert_logistic_reg_model.joblib`</li>
            </ul>
          </ul>
        </div>

        <div>
          <h3 className="text-3xl font-bold mb-4">5. Conclusion</h3>
          <p className="text-lg leading-relaxed">
            This project successfully demonstrated the process of building a sentiment analysis model
            for Twitter data. Both the traditional machine learning approach (TF-IDF with Logistic
            Regression) and the deep learning approach (BERT) were explored.
          </p>
          <ul className="list-disc list-inside text-lg leading-relaxed ml-4">
            <li>The TF-IDF with Logistic Regression model, after handling class imbalance, provided a
                good baseline and performed reasonably well with an accuracy of 84%.</li>
            <li>The BERT-based model also achieved a high accuracy of 83%, showcasing the
                effectiveness of using pre-trained language models for NLP tasks.</li>
          </ul>
          <p className="text-lg leading-relaxed mt-4">
            The saved models can be used to predict the sentiment of new tweets.
          </p>
        </div>

      </div>
    </section>
  );
};

export default Report;
