from transformers import AutoModelForMaskedLM, AutoTokenizer

from obfs.strategy import (
    random_mask_perplexity,
    random_mask_perplexity_batch,
    unk_token_ratio,
)

model_name = "./models/distilbert/distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "20% Off Ghost Drive | #1 Customer Fave"

print(f"unk_token_ratio: {unk_token_ratio(text, tokenizer)}")
print(
    f"random_mask_perplexity: {random_mask_perplexity(text, tokenizer, model, debug=True, seed=42)}"
)
print(
    f"random_mask_perplexity_batch: {random_mask_perplexity_batch(text, tokenizer, model, n_trials=3)}"
)
