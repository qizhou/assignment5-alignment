from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import json
import torch


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-Math-1.5B")

trainfile = "./data/math/sft.jsonl"
prompts_strs = []
output_strs = []
with open(trainfile) as f:
    for line in f:
        j = json.loads(line)
        prompts_strs.append(j["problem"])
        output_strs.append(j["reasoning_trace"])



# gradient_accumulation_steps = 4
# for idx, (inputs, labels) in enumerate(data_loader):
#     # Forward pass.
#     logits = model(inputs)
#     loss = loss_fn(logits, labels) / gradient_accumulation_steps
#     # Backward pass.
#     loss.backward()
#     if (idx + 1) % gradient_accumulation_steps == 0:
#         # Update weights every `gradient_accumulation_steps` batches.
#         optimizer.step()

#         # Zero gradients every `gradient_accumulation_steps` batches.
#         optimizer.zero_grad()