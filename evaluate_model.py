from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model():
    # Load the trained model and vectorizer
    model = joblib.load('resume_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')

    # Load and preprocess the test data
    test_data = pd.read_csv(
        '/home/user/Desktop/Projects/ResumeCategorizerByTahsinZaman/categorized_resumes.csv')
    X_test = test_data['filename'].tolist()
    y_test = test_data['category']

    # Transform the test data using the same vectorizer used during training
    X_test_vec = vectorizer.transform(X_test)

    # Predict on the test set
    y_test_pred = model.predict(X_test_vec)

    # Convert labels to consistent format
    if y_test.dtype == 'object':
        y_test = encoder.transform(y_test)
    if isinstance(y_test_pred[0], str):
        y_test_pred = encoder.transform(y_test_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_test_pred)

    # Print evaluation metrics
    print("Test Classification Report:")
    report = classification_report(y_test, y_test_pred, target_names=encoder.classes_, zero_division=0)
    print(report)

    # Print accuracy separately
    print(f"Accuracy: {accuracy:.4f}")

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, display_labels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot to a file
    print("Confusion matrix saved as 'confusion_matrix.png'.")


if __name__ == "__main__":
    evaluate_model()
