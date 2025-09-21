import joblib
import os
import re
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.linear_model import LogisticRegression

class TfidfTweetClassifier:
    def __init__(self, model_path, vectorizer_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF Vectorizer file not found at: {vectorizer_path}")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def preprocess_text(self, text):
        # Basic text cleaning: lowercasing, removing special characters
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def predict(self, tweet_text):
        preprocessed_text = self.preprocess_text(tweet_text)
        vectorized_text = self.vectorizer.transform([preprocessed_text])
        prediction = self.model.predict(vectorized_text)
        return prediction[0]

class BertTweetClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model = joblib.load(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', use_safetensors=False)

    def predict(self, tweet_text):
        tokenized = self.tokenizer.encode(tweet_text, add_special_tokens=True, max_length=128, truncation=True, padding='max_length')
        
        padded = np.array([tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        
        input_ids = tf.convert_to_tensor(padded)
        attention_mask_tensor = tf.convert_to_tensor(attention_mask)
        
        with tf.device('/CPU:0'):
            outputs = self.bert_model(input_ids, attention_mask=attention_mask_tensor)
        
        last_hidden_states = outputs[0]
        cls_embeddings = last_hidden_states[:, 0, :].numpy()
        
        prediction = self.model.predict(cls_embeddings)
        return prediction[0]