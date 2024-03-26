# Load model directly
import datasets
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from seminar_02.train import IMDbDataset, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_size = 256
batch_size = 64


def main():
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")

    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    # model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=4)

    dataset = datasets.load_dataset('ag_news')

    val_encodings = tokenizer(dataset['test']['text'],
                              truncation=True,
                              padding=True,
                              max_length=seq_size,
                              return_tensors='pt').to(device)
    val_dataset = IMDbDataset(val_encodings, dataset['test']['label'])
    eval_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size)

    model = model.to(device)
    evaluate(model, eval_dataloader, 0)


if __name__ == '__main__':
    main()
