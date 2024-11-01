from transformers import AutoModelForMaskedLM, AutoTokenizer

from strategy import unk_token_ratio

model_name = "../models/distilbert/distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "20% Off Ghost Drive | #1 Customer Fave"

print(f"unk_token_ratio: {unk_token_ratio(text, tokenizer)}")
