from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    sample = 'Привет мир!!!'
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-fr")

    ids = tokenizer(sample, return_tensors='pt')

    output = model.generate(**ids)

    text = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(text)


if __name__ == '__main__':
    main()
