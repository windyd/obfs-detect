import random
from copy import deepcopy

import torch
from rich import print
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


# Initialize the model and tokenizer
# model_name = 'nghuyong/ernie-3.0-nano-zh' # Not LM
# model_name = "distilbert/distilbert-base-multilingual-cased"
model_name = "../models/distilbert/distilbert-base-multilingual-cased"
# model_name = 'openai-community/gpt2' # 548M


tokenizer = AutoTokenizer.from_pretrained(model_name)
if "gpt" in model_name:
    model = AutoModelForCausalLM.from_pretrained(model_name)
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name)


def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    labels = inputs.input_ids

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()


def perplexity_hf(text, model, tokenizer, stride: int = 250):
    max_length = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings"
    )
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        # end_loc = min(begin_loc + max_length, seq_len)
        end_loc = min(begin_loc + max_length, seq_len, begin_loc + stride)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        # input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        input_ids = encodings.input_ids[:, begin_loc:end_loc]

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore indices

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


text = "尊敬的企箱用户xxxxx@abaesc.com 你好由于系统储存容量满载超"
# masked_text = "尊敬的企箱用户xxxxx@abaesc.com 你好由于系统[MASK]容量满载超"
obfs_text = "尊敬<喘挂续秒备讲脾>的<苗好一半谷，妻好一半福。>企<仙旋谓怜淇快敦莞雍渎>业<肥茧桂冻镌郊解麟>邮<盗马贼披袈裟嫁祸于人 >箱<老驴打滚翻不过身来 >用<廓堵灌臂契骂群怨荻利>户<若你有一种思想.我也有一种思想,>xxxxx@abaesc.com 你好由于<山洪冲石子不滚也得滚>系<滴式纭王毋猎>统<个性和魅力……是学不会，装不像的。>储存<蜚晓西雀辆>容量满<歇后语投机取巧类>载超<理智可以制定法律来约束感情,可是热情激动起来,就会把冷酷"
# text = obfs_text


# Example usage
def test_last_token_mask():
    perplexity = perplexity_hf(text, model, tokenizer, stride=10)
    obfs_perplexity = perplexity_hf(obfs_text, model, tokenizer, stride=10)

    print(f"Perplexity: {perplexity}")
    print(f"obfs_Perplexity: {obfs_perplexity}")

    perplexity = calculate_perplexity(text)
    obfs_perplexity = calculate_perplexity(obfs_text)

    print(f"Perplexity: {perplexity}")
    print(f"obfs_Perplexity: {obfs_perplexity}")
