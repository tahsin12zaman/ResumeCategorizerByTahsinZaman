from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from data_preprocessing import preprocess_data


def train_model():
    (X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer) = preprocess_data('/home/user/Desktop/Projects/ResumeCategorizerByTahsinZaman/dataset/data')

    # Train a Support Vector Machine model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_val_pred = model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=encoder.classes_))

    # Save the model and vectorizer
    joblib.dump(model, 'resume_classifier_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')


if __name__ == "__main__":
    train_model()
