import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    max_len = 0
    mask = []
    ids = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]

        max_len = max(len(output_ids) + len(prompt_ids), max_len)
        # the rest (if have) of mask will be padded as False
        mask.append([i >= len(prompt_ids) - 1 for i in range(len(prompt_ids) + len(output_ids) - 1)])
        prompt_ids.extend(output_ids)
        ids.append(prompt_ids)

    # pad zero and mask
    for i in range(len(ids)):
        # padding ids
        while len(ids[i]) < max_len:
            ids[i].append(tokenizer.pad_token_id)
        # padding mask
        while len(mask[i]) < max_len - 1:
            mask[i].append(False)

    # obtain input_ids and labels
    input_ids = []
    labels = []
    for i in range(len(ids)):
        input_ids.append(ids[i][:-1])
        labels.append(ids[i][1:])

    x = torch.tensor(input_ids)

    return {"input_ids": torch.tensor(input_ids), "labels": labels, "response_mask": mask}


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


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    s = (tensor * mask).sum(dim) / normalize_constant
    return s


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch = policy_log_probs.shape[0]
    # TODO: why not average over sequence?
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant) / gradient_accumulation_steps / batch
    loss.backward()
    return loss, {}
