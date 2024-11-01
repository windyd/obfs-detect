import random
from copy import deepcopy
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def random_mask_test(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    seed: Optional[int] = None,
):
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    # randomly select a token to mask
    masked_inputs = deepcopy(inputs)
    if seed:
        random.seed(seed)
    masked_pos = random.randint(0, len(inputs.input_ids[0]) - 1)
    masked_inputs.input_ids[0, masked_pos] = tokenizer.mask_token_id
    masked_text = tokenizer.decode(masked_inputs.input_ids[0])

    with torch.no_grad():
        outputs = model(**masked_inputs)
        predictions = outputs.logits

    MASK_TOKEN_IN_CHAR = tokenizer.decode([tokenizer.mask_token_id])
    mask_token_index = (masked_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
        as_tuple=True
    )[0]
    predicted_token_ids = predictions[0, mask_token_index].argmax(axis=-1)
    probs = predictions[0, mask_token_index]
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
    # completed_text = text.replace("[MASK]", predicted_tokens[0])
    completed_text = masked_text.replace(MASK_TOKEN_IN_CHAR, predicted_tokens[0])

    print(f"original: {tokenizer.decode(tokenizer.encode_plus(text).input_ids)}")
    print(f"masked: {masked_text}")
    print(f"completed_text: {completed_text}")
    print(f"probs: {probs}")

    top_k = 5
    top_k_tokens = torch.topk(probs, top_k).indices
    print(f"top_k_tokens: {top_k_tokens}")
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_tokens[0])
    print(f"top_k_tokens: {top_k_tokens}")
    normalized_probs = torch.nn.functional.softmax(probs, dim=-1)
    print(f"normalized_probs: {normalized_probs}")
    top_k_probs = normalized_probs.topk(top_k).values
    print(f"top_k_probs: {top_k_probs}")
    return top_k_probs


def random_mask_perplexity(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    debug: bool = False,
    seed: Optional[int] = None,
):
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    # randomly select a token to mask
    masked_inputs = deepcopy(inputs)
    if seed:
        random.seed(seed)
    masked_pos = random.randint(0, len(inputs.input_ids[0]) - 1)
    masked_inputs.input_ids[0, masked_pos] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**masked_inputs, labels=masked_inputs.input_ids)
        # NOTE: code under the hood
        # masked_lm_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

    loss = outputs.loss
    perplexity = torch.exp(loss)

    if debug:
        masked_text = tokenizer.decode(masked_inputs.input_ids[0])
        print(f"original: {tokenizer.decode(inputs.input_ids[0])}")
        print(f"masked: {masked_text}")

    return perplexity.item()
