import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import PyPDF2


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def categorize_resumes(directory):
    # Load the model and vectorizer
    model = joblib.load('resume_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')

    # Prepare output CSV
    categorized_resumes = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                text_vec = vectorizer.transform([text])
                prediction = model.predict(text_vec)
                category = encoder.inverse_transform(prediction)[0]

                # Move file to category folder
                category_folder = os.path.join(directory, category)
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)
                new_file_path = os.path.join(category_folder, file)
                os.rename(file_path, new_file_path)

                # Record the result
                categorized_resumes.append({'filename': file, 'category': category})

    # Save results to CSV
    df = pd.DataFrame(categorized_resumes)
    df.to_csv('categorized_resumes.csv', index=False)


if __name__ == "__main__":
    import sys

    directory_path = sys.argv[1]
    categorize_resumes(directory_path)
