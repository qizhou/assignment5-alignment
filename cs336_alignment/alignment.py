import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # Numerical stability: subtract max
    logits = logits - logits.max(dim=2, keepdim=True)[0]

    # Probabilities
    probs = torch.softmax(logits, dim=2)

    # Log-probabilities using LogSumExp trick
    log_probs = logits - torch.logsumexp(logits, dim=2, keepdim=True)

    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=2)
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # return (batch_size, sequence_length, vocab_size)
    logits = model(input_ids).logits
    max_v, _ = logits.max(2, keepdim=True)

    # auto broadcast
    ev = (logits - max_v).exp()
    sum_ev = ev.sum(2)

    # return (batch_size, sequence_length)
    ev = torch.gather(ev, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)

    # obtain log prob
    logp = (ev/sum_ev).log()

    if return_token_entropy:
        entropy = compute_entropy(logits)
    else:
        entropy = None
    return {"log_probs": logp, "token_entropy": entropy}