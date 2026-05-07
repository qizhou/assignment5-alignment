from alignment import tokenize_prompt_and_output
from transformers import AutoTokenizer
import json
import torch

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-Math-1.5B")

promptfile = "./cs336_alignment/prompts/r1_zero.prompt"
trainfile = "./data/math/sft.jsonl"
outputfile = "./sft_tokens.ckp"

with open(promptfile, "r", encoding="utf-8") as f:
    prompt_template = f.read()

prompts_strs = []
output_strs = []
with open(trainfile) as f:
    for line in f:
        j = json.loads(line)
        prompts_strs.append(prompt_template.format(question=j["problem"]))
        output_strs.append(j["reasoning_trace"])

        # print(prompts_strs[-1] + output_strs[-1])


ret = tokenize_prompt_and_output(prompts_strs, output_strs, tokenizer)

torch.save(ret, outputfile)