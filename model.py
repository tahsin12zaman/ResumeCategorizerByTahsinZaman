from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch


def train_model(X_train, y_train, X_val, y_val):
    # Define the model and training arguments
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y_train)))
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        eval_dataset=torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    )

    # Train the model
    trainer.train()

    return model, trainer


def save_model(model, file_path='bert_resume_categorizer_model.pt'):
    # Save the trained model
    model.save_pretrained(file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path='bert_resume_categorizer_model.pt'):
    # Load the saved model
    model = BertForSequenceClassification.from_pretrained(file_path)
    return model
