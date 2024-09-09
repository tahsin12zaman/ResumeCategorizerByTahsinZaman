import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import PyPDF2


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def preprocess_data(root_dir):
    data = []
    labels = []
    categories = os.listdir(root_dir)

    for category in categories:
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            pdf_files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
            for pdf_file in pdf_files:
                pdf_path = os.path.join(category_path, pdf_file)
                text = extract_text_from_pdf(pdf_path)
                data.append(text)
                labels.append(category)

    df = pd.DataFrame({'text': data, 'label': labels})
    X = df['text']
    y = df['label']

    # Splitting the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Label Encoding
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    return (X_train_vec, X_val_vec, X_test_vec, y_train_enc, y_val_enc, y_test_enc, encoder, vectorizer)


# Example usage
root_directory = '/home/user/Desktop/Projects/ResumeCategorizerByTahsinZaman/dataset/data'
(X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer) = preprocess_data(root_directory)
