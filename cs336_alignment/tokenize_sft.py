from alignment import tokenize_prompt_and_output
from transformers import AutoTokenizer
import json
import torch

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-Math-1.5B")

trainfile = "./data/math/sft.jsonl"
outputfile = "./sft_tokens.ckp"
prompts_strs = []
output_strs = []
with open(trainfile) as f:
    for line in f:
        j = json.loads(line)
        prompts_strs.append(j["problem"])
        output_strs.append(j["reasoning_trace"])


ret = tokenize_prompt_and_output(prompts_strs, output_strs, tokenizer)

torch.save(ret, outputfile)