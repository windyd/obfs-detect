from transformers import AutoTokenizer


def unk_token_cnt(text: str, tokenizer: AutoTokenizer):
    """
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> text = "this is an [UNK] token and another [UNK] token"
    >>> print(
            count_unk_tokens(text, tokenizer)
        )  # Output will depend on the tokenizer's behavior
    """
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Initialize the counter
    unk_count = 0

    # Count the unknown tokens directly using a list comprehension
    unk_count = sum(1 for token in tokens if token == tokenizer.unk_token)

    # Return the count of unknown tokens
    return unk_count


def unk_token_ratio(text: str, tokenizer: AutoTokenizer):
    """
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> text = "this is an [UNK] token and another [UNK] token"
    >>> print(
            unk_token_ratio(text, tokenizer)
        )  # Output will depend on the tokenizer's behavior
    """
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Count the unknown tokens
    unk_count = sum(1 for token in tokens if token == tokenizer.unk_token)

    # Calculate the ratio of unknown tokens to total tokens
    total_tokens = len(tokens)
    unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0

    # Return the ratio of unknown tokens
    return unk_ratio
