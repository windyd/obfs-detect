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


def random_mask(
    tokenized_input,
    mask_token_id=103,
    pad_token_id=0,
):
    # Calculate real length excluding padding, BOS, and EOS tokens
    valid_tokens_mask = tokenized_input != pad_token_id
    real_length = valid_tokens_mask.sum(dim=-1, dtype=torch.float32)

    # Note: We need to make sure that:
    # 1. the first and last tokens are not masked
    # 2. shift the random position by 1 to the right
    rand_pos = 1 + (torch.rand_like(real_length) * (real_length - 2)).to(
        torch.int64
    )  # torch.int64 is needed for scatter_
    masked_input = tokenized_input.clone()
    # tokenized_input_clone[rand_pos] = mask_token_id # this is incorrect
    masked_input.scatter_(-1, rand_pos.unsqueeze(-1), mask_token_id)
    return masked_input, rand_pos


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


def random_mask_perplexity_batch(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    debug: bool = False,
    n_trials: int = 1,
):
    original_inputs = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings,
    )
    original_inputs_ids = original_inputs.input_ids.repeat(n_trials, 1)
    masked_inputs_ids = original_inputs_ids.clone()
    masked_inputs_ids, mask_pos = random_mask(
        masked_inputs_ids,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model(masked_inputs_ids, labels=original_inputs_ids)
        # NOTE: code under the hood
        # masked_lm_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    if debug:
        masked_text = list(map(lambda x: tokenizer.decode(x), masked_inputs_ids))
        print(masked_text)

    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity


def random_mask_bernoulli(
    tokenized_input, mask_token_id=103, mask_prob=0.15, pad_token_id=0
):
    """
    Randomly mask tokens in the tokenized input with guarantees. The random mask is
    generated using a Bernoulli distribution.

    Args:
        tokenized_input (torch.Tensor): The tokenized id input tensor.
        mask_token_id (int): The ID of the mask token.
        mask_prob (float): The probability of masking each token.
        pad_token_id (int): The ID of the padding token.

    Returns:
        torch.Tensor: The masked tokenized input tensor.
    """

    def mask_once(t_input):
        nonlocal mask_prob, mask_token_id, pad_token_id
        # Mask prob
        mask_prob_tensor = torch.full(t_input.shape, mask_prob)

        # Set the probability of masking the padding tokens to 0
        mask_prob_tensor = torch.where(t_input == pad_token_id, 0, mask_prob)

        # Generate random mask positions
        mask_positions = torch.bernoulli(mask_prob_tensor).bool()

        # Get the rows without random mask
        rows_without_mask = torch.where(~torch.any(mask_positions, dim=-1))[0]
        return mask_positions, rows_without_mask

    # Regenerate the mask positions until all rows have at least one mask
    mask_positions, rows_without_mask = mask_once(tokenized_input)
    while torch.any(rows_without_mask):
        mask_positions, rows_without_mask = mask_once(tokenized_input)

    # Create a copy of the tokenized input to apply the mask
    masked_input = tokenized_input.clone()

    # Apply the mask
    masked_input[mask_positions] = mask_token_id

    return masked_input
