import os
import torch
from transformers import BertTokenizer
from model import load_model
import pandas as pd
from preprocessing import tokenize_text

# Load the model and tokenizer
model_path = 'bert_resume_categorizer_model.pt'
model = load_model(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def read_resume_file(file_path):
    """Attempt to read a resume file with multiple encodings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                return f.read()
        except UnicodeDecodeError:
            print(f"Could not read file {file_path} due to encoding issues.")
            return None  # Returning None if both encodings fail

def process_resumes(input_dir):
    categorized_resumes = []
    # Load the preprocessed data for prediction
    for category_folder in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category_folder)
        if os.path.isdir(category_path):
            for resume_file in os.listdir(category_path):
                resume_path = os.path.join(category_path, resume_file)
                resume_text = read_resume_file(resume_path)

                if resume_text is None:
                    continue

                # Clean and tokenize the resume text
                cleaned_resume = preprocessing.clean_text(resume_text)
                encodings = tokenizer(cleaned_resume, truncation=True, padding=True, max_length=512, return_tensors='pt')

                # Predict the category
                with torch.no_grad():
                    outputs = model(**encodings)
                    predicted_class = torch.argmax(outputs.logits, dim=1).item()

                # Create a folder for the predicted category
                output_folder = os.path.join(input_dir, str(predicted_class))
                os.makedirs(output_folder, exist_ok=True)
                shutil.move(resume_path, os.path.join(output_folder, resume_file))

                # Add to the categorized resumes list for CSV generation
                categorized_resumes.append([resume_file, predicted_class])

    # Save the categorized resumes to a CSV file
    output_csv_path = 'categorized_resumes.csv'
    df = pd.DataFrame(categorized_resumes, columns=['filename', 'category'])
    df.to_csv(output_csv_path, index=False)
    print(f"Categorized resumes saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_resume_directory>")
        sys.exit(1)

    resume_dir = sys.argv[1]
    process_resumes(resume_dir)
