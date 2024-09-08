import preprocessing
import model
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Load and preprocess data
resume_df = preprocessing.load_data('/home/user/Downloads/2/dataset/Resume/Resume.csv')
resume_df = preprocessing.preprocess_data(resume_df)

# Tokenize and encode text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = preprocessing.tokenize_text(resume_df, tokenizer)
X = encodings['input_ids']
y = resume_df['Category'].values

# Split data
X_train, X_val, X_test, y_train, y_val, y_test, attention_masks = preprocessing.split_data(encodings, y)

# Train the model
model, trainer = model.train_model(X_train, y_train, X_val, y_val)

# Save the model
model.save_model()

# Evaluate the model
trainer.evaluate()
