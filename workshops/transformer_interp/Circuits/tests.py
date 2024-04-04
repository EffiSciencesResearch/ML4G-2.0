import torch as t
from torch import Tensor
from typing import List, Tuple, Dict, Any, Union
from jaxtyping import Float, Int, Bool
from transformer_lens import utils, HookedTransformer, ActivationCache


def logit_attribution(
    tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    cache: ActivationCache,
    token_position: int,
) -> Float[Tensor, "layers heads"]:
    """
    Computes the logit attribution for a specific token position in the input sequence.

    Args:
        tokens (Int[Tensor, "batch seq"]): The input token IDs tensor with shape (batch_size, sequence_length).
        model (HookedTransformer): The HookedTransformer model instance.
        cache (ActivationCache): The activation cache containing the intermediate results.
        token_position (int): The position of the token in the input sequence for which to compute the attribution.

    Returns:
        Float[Tensor, "layers heads"]: The logit attribution tensor with shape (num_layers, num_heads).

    Description:
        This function computes the logit attribution for a specific token position in the input sequence.
        It unembeds the output of each attention head in each layer, and sees what upweight it gives on the correct next token.

    Note:
        - The input `tokens` tensor is assumed to have a batch size of 1.
        - The `token_position` is zero-indexed, meaning the first token in the sequence has a position of 0.
        - The returned attention pattern has shape (num_layers, num_heads), representing the attribution scores
          for each layer and attention head.
    """
    # Retrieve the attention results from the activation cache for each transformer block
    results = [cache[f"blocks.{i}.attn.hook_result"] for i in range(len(model.blocks))]

    # Stack the attention results along the layer dimension
    results = t.stack(results, dim=1)

    # Select the attention results corresponding to the specified token position
    results = results[token_position, :, :, :]

    # Pass the selected attention results through the model's unembed function to obtain the logits
    logits = model.unembed(results)

    # Get the ID of the next token in the sequence
    next_token_id = tokens[0, token_position + 1]

    # Extract the logits corresponding to the next token ID
    attention_pattern = logits[:, :, next_token_id]

    return attention_pattern


def test_logit_attribution(implementation, model):
    test_string = "The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(test_string)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    attention_pattern = logit_attribution(tokens, model, cache, token_position=3)
    comparison_pattern = implementation(tokens, model, cache, token_position=3)

    assert t.allclose(attention_pattern, comparison_pattern)
    print("Logit attribution test passed!")


def average_over_condition(tensor, condition):
    I, J, K = tensor.shape
    return [
        sum(tensor[i, j, k] for j in range(J) for k in range(K) if condition(j, k))
        / sum(condition(j, k) for j in range(J) for k in range(K))
        for i in range(I)
    ]


def over_threshhold_attn(cache, condition, threshhold=0.5, sorce="pattern"):
    return_values = []

    for layer, pattern in enumerate(cache.stack_activation("pattern")):
        scores = average_over_condition(pattern, condition)
        indices = [i for i, s in enumerate(scores) if s > threshhold]
        for i in indices:
            return_values.append(f"L{layer+1}H{i}")
    return return_values


def current_attn_detector(cache: ActivationCache, threshhold=0.3) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """

    def cond(i, j):
        return i == j

    return over_threshhold_attn(cache, cond, threshhold=threshhold)


def prev_attn_detector(cache: ActivationCache, threshhold=0.3) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """

    def cond(i, j):
        return i - j == 1

    return over_threshhold_attn(cache, cond, threshhold=threshhold)


def first_attn_detector(cache: ActivationCache, threshhold=0.3) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """

    def cond(i, j):
        return j == 0

    return over_threshhold_attn(cache, cond, threshhold=threshhold)


def find_repeating_rows(tensor):
    """
    Finds repeating rows (vectors) in a 2D torch tensor.

    Args:
    tensor (torch.Tensor): A 2D torch tensor.

    Returns:
    dict: A dictionary where keys are the indices of repeating rows,
          and values are the indices where those rows last occurred.
    """
    last_occurrence = {}
    repeats = {}

    for pos, token in enumerate(tensor[0]):
        id = token.item()

        if id in last_occurrence:
            repeats[pos] = last_occurrence[id]
        last_occurrence[id] = pos

    return repeats


def induction_attn_detector(
    cache: ActivationCache, tokens, off_by_one=True, threshhold=0.3
) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    repeat_dict = find_repeating_rows(t.tensor(tokens))

    def cond(i, j):
        if i not in repeat_dict.keys():
            return False
        to_add = 1 if off_by_one else 0
        return repeat_dict[i] + to_add == j

    return over_threshhold_attn(cache, cond, threshhold=threshhold)


def test_average_over_condition(implementations):
    tensor = t.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]]
    )
    condition = lambda i, j: i == j
    result = average_over_condition(tensor, condition)
    comparison = implementations(tensor, condition)
    assert t.allclose(result, comparison)
    print("Average over condition test passed!")


def test_current_attn_detector(implementations, model):
    test_string = "The quick brown fox jumps over the lazy dog.The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(test_string)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    result = current_attn_detector(cache)
    comparison = implementations(cache)
    assert result == comparison
    print("Current attention detector test passed!")


def test_prev_attn_detector(implementations, model):
    test_string = "The quick brown fox jumps over the lazy dog.The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(test_string)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    result = prev_attn_detector(cache)
    comparison = implementations(cache)
    assert result == comparison
    print("Previous attention detector test passed!")


def test_first_attn_detector(implementations, model):
    test_string = "The quick brown fox jumps over the lazy dog.The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(test_string)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    result = first_attn_detector(cache)
    comparison = implementations(cache)
    assert result == comparison
    print("First attention detector test passed!")


def test_induction_attn_detector(implementations, model):
    test_string = "The quick brown fox jumps over the lazy dog.The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(test_string)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    result = induction_attn_detector(cache, tokens)
    comparison = implementations(cache, tokens)
    assert result == comparison
    print("Induction attention detector test passed!")


def test_induction_attn_detector(implementations, model):
    test_string = "The quick brown fox jumps over the lazy dog.The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(test_string)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    result = induction_attn_detector(cache, tokens)
    comparison = implementations(cache, tokens)
    assert result == comparison
    print("Induction attention detector test passed!")
