import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess_data(df):
    df['cleaned_resume'] = df['Resume_str'].apply(clean_text)
    return df

def tokenize_text(df, tokenizer):
    # Tokenize and encode the text
    encodings = tokenizer(df['cleaned_resume'].tolist(), truncation=True, padding=True, max_length=512)
    return encodings

def split_data(encodings, labels):
    # Create a dataset with encoded inputs and labels
    inputs = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    X_train, X_temp, y_train, y_temp = train_test_split(inputs, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, attention_masks
