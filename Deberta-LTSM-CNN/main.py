import torch
from transformers import DebertaTokenizer, DebertaModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaTokenizer

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
deberta_model = DebertaModel.from_pretrained('microsoft/deberta-large')

sample_text = "This is a sample sentence for testing the DeBERTa model."
inputs = tokenizer(sample_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)


class HybridDeBERTaModel(nn.Module):
    def __init__(self, pretrained_model=None, lstm_hidden_size=128, cnn_filters=64):
        super(HybridDeBERTaModel, self).__init__()
        # Use the provided DeBERTa model
        self.deberta = pretrained_model if pretrained_model else DebertaModel.from_pretrained('microsoft/deberta-large')
        hidden_size = self.deberta.config.hidden_size  # 1024 for deberta-large

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=lstm_hidden_size,
                            bidirectional=True,
                            batch_first=True)

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=cnn_filters, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=cnn_filters, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=cnn_filters, kernel_size=4)
        self.pool = nn.AdaptiveMaxPool1d(1)  # Adaptive pooling for dynamic sequence length

        # Fully Connected Layer
        self.fc = nn.Linear(lstm_hidden_size * 2 + cnn_filters * 3, 5)  # 5 labels
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # DeBERTa Embeddings
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

        # LSTM
        lstm_out, _ = self.lstm(hidden_states)  # Shape: [batch_size, sequence_length, lstm_hidden_size * 2]
        lstm_out = lstm_out[:, -1, :]  # Use the last hidden state: [batch_size, lstm_hidden_size * 2]

        # CNN
        cnn_in = hidden_states.permute(0, 2, 1)  # Transpose to [batch_size, hidden_size, sequence_length]
        conv1_out = self.pool(torch.relu(self.conv1(cnn_in))).squeeze(-1)  # Shape: [batch_size, cnn_filters]
        conv2_out = self.pool(torch.relu(self.conv2(cnn_in))).squeeze(-1)
        conv3_out = self.pool(torch.relu(self.conv3(cnn_in))).squeeze(-1)

        # Combine LSTM and CNN outputs
        combined = torch.cat((lstm_out, conv1_out, conv2_out, conv3_out),
                             dim=1)  # Shape: [batch_size, lstm_hidden_size * 2 + cnn_filters * 3]

        logits = self.fc(combined)  # Shape: [batch_size, 5]
        return self.sigmoid(logits)


hybrid_model = HybridDeBERTaModel(pretrained_model=deberta_model)
hybrid_model.to('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_dataset(data, text_column, label_columns, max_length=128):

    texts = data[text_column].tolist()
    labels = data[label_columns].values

    # Tokenize the text data
    tokenized_texts = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    return {
        'input_ids': tokenized_texts['input_ids'],
        'attention_mask': tokenized_texts['attention_mask'],
        'labels': torch.tensor(labels, dtype=torch.float)
    }


data = pd.read_csv('data/aug_train.csv')
text_column = 'text'
label_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise']

processed_data = preprocess_dataset(data, text_column, label_columns)

torch.save(processed_data, 'processed_data.pt')


processed_data = torch.load('processed_data.pt')

class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

dataset = SentimentDataset(
    input_ids=processed_data['input_ids'],
    attention_mask=processed_data['attention_mask'],
    labels=processed_data['labels']
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_data = pd.read_csv("data/test_eng.csv")


test_processed_data = preprocess_test_data(test_data, text_column='text', tokenizer=tokenizer)

test_dataset = SentimentDataset(
    input_ids=test_processed_data['input_ids'],
    attention_mask=test_processed_data['attention_mask']
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

hybrid_model.eval()
test_predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = hybrid_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = (outputs > 0.5).int()  # Convert probabilities to binary predictions
        test_predictions.extend(preds.cpu().numpy())

LABEL_COLUMNS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
submission_df = pd.DataFrame(test_predictions, columns=LABEL_COLUMNS)
submission_df['id'] = test_data['id']  # Ensure 'id' column matches test dataset

submission_df = submission_df[['id'] + LABEL_COLUMNS]

submission_file_path = "submission.csv"
submission_df.to_csv(submission_file_path, index=False)
print(f"Submission file saved to {submission_file_path}")
