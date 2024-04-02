import torch as t
from einops import repeat


def logit_lense_analysis(input_prompt: str, gpt2_small):
    """
    Performs logit lens analysis on the given input prompt.

    Args:
        input_prompt (str): The input prompt to analyze.

    Returns:
        tuple: A tuple containing the following:
            - tokens (list): The tokenized input prompt.
            - top_unembedded_residual_stream_caches (torch.Tensor): The top unembedded residual stream caches.
            - kl_divergences (torch.Tensor): The KL divergences between the unembedded residual stream caches and the final logits.
    """
    # Tokenize the input prompt
    tokens = gpt2_small.to_tokens(input_prompt)

    # Run the model with caching
    final_logits, cache = gpt2_small.run_with_cache(tokens, remove_batch_dim=True)

    # Get the number of layers in the model
    num_layers = len(gpt2_small.blocks)

    # Extract the residual stream caches from the cache
    residual_stream_caches = [
        cache[f"blocks.{i}.ln2.hook_normalized"] for i in range(num_layers)
    ]

    # Stack the residual stream caches along the layer dimension
    residual_stream_caches = t.stack(residual_stream_caches, dim=1)

    # Unembed the residual stream caches
    unembedded_residual_stream_caches = gpt2_small.unembed(residual_stream_caches)

    # Transform the unembedded residual stream caches into log probabilities
    unembedded_residual_stream_caches = unembedded_residual_stream_caches.log_softmax(
        dim=-1
    )

    # Get the top unembedded residual stream caches
    top_unembedded_residual_stream_caches = t.argmax(
        unembedded_residual_stream_caches, dim=-1
    )

    # Create a tensor of the same shape as the final logits of the model at each position
    final_logits = repeat(
        final_logits[0], "nb_tokens vocab -> nb_tokens layer vocab", layer=num_layers
    )

    # Convert the final logits to probabilities
    final_probs = final_logits.softmax(dim=-1)

    # Calculate the KL divergences between the unembedded residual stream caches and the final logits
    kl_divergences = t.nn.functional.kl_div(
        unembedded_residual_stream_caches, final_probs, reduction="none"
    ).sum(-1)

    return tokens, top_unembedded_residual_stream_caches, kl_divergences


def test_logit_lense_analysis(implementation_function, model):
    input_prompt = "The quick brown fox jumps over the lazy dog."
    tokens, top_unembedded_residual_stream_caches, kl_divergences = (
        implementation_function(input_prompt)
    )

    (
        comparison_tokens,
        comparison_top_unembedded_residual_stream_caches,
        comparison_kl_divergences,
    ) = logit_lense_analysis(input_prompt, model)

    assert t.allclose(tokens, comparison_tokens)
    assert t.allclose(
        top_unembedded_residual_stream_caches,
        comparison_top_unembedded_residual_stream_caches,
        atol=1e-3,
    )
    assert t.allclose(kl_divergences, comparison_kl_divergences, atol=1e-3)

    print("Logit lense analysis test passed")


def prompt_to_residual_stream_activations(prompt, demarcation_token, gpt2_small):
    """
    Computes the residual stream activations for the first token after the demarkation token in the prompt.

    Args:
        prompt (str): The input prompt string.
        demarcation_token (int): The token ID representing the demarcation point in the prompt.

    Returns:
        torch.Tensor: The residual stream activations for the final token in the prompt.
    """
    # Tokenize the prompt and convert it to a list
    tokens = gpt2_small.to_tokens(prompt)[0].tolist()

    # Find the position of the demarcation token in the tokenized prompt
    demarcation_position = tokens.index(demarcation_token)

    # Truncate the tokens up to the demarcation token and include two additional tokens
    tokens = tokens[: demarcation_position + 2]

    # Convert the tokens to a tensor and add a batch dimension
    tokens = t.tensor(tokens).unsqueeze(0)

    # Run the model with the truncated tokens and obtain the final logits and cache
    final_logits, cache = gpt2_small.run_with_cache(tokens, remove_batch_dim=True)

    # Get the number of layers in the model
    num_layers = len(gpt2_small.blocks)

    # Extract the residual stream activations from the cache for each layer
    residual_stream_caches = [
        cache[f"blocks.{i}.ln2.hook_normalized"] for i in range(num_layers)
    ]

    # Stack the residual stream activations along a new dimension
    residual_stream_caches = t.stack(residual_stream_caches, dim=1)

    # Extract the residual stream activations for the final token
    final_token_residual_stream_caches = residual_stream_caches[-1]

    return final_token_residual_stream_caches


def test_prompt_to_residual_stream_activations(implementation_function, model):
    prompt = "The quick brown fox jumps over the lazy : dog."

    demarkation_token = model.tokenizer.encode(" :")[-1]
    residual_stream_activations = implementation_function(prompt, demarkation_token)

    comparison_residual_stream_activations = prompt_to_residual_stream_activations(
        prompt, demarkation_token, model
    )

    assert t.allclose(
        residual_stream_activations, comparison_residual_stream_activations, atol=1e-3
    )

    print("Prompt to residual stream activations test passed")
