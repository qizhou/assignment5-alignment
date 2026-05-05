from typing import Callable, List

from vllm import LLM, SamplingParams
import json
import drgrpo_grader


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    correct = 0
    format_correct = 0 # answer is 0
    incorrect = 0
    js = []
    for output, answer in zip(outputs, answers):
        # print(output.outputs[0].text)
        result = reward_fn(output.outputs[0].text, answer)
        # print(result)
        correct += result["reward"] # both format and answer
        format_correct += result["format_reward"] * (1 - result["answer_reward"]) # format and !answer
        incorrect += (1 - result["format_reward"]) * (1 - result["answer_reward"])
        js.append({"result": result, "actual": output.outputs[0].text})

    with open("result.jsonl", "w") as f:
        for j in js:
            json.dump(j, f)
            f.write("\n")
    print(f"Total {len(answers)}, correct {correct}, format_correct {format_correct}, incorrect {incorrect}")


# datafile = "./data/gsm8k/test.jsonl"
# answer_key = "answer"
# question_key = "question"
datafile = "./data/math/val.jsonl"
answer_key = "expected_answer"
question_key = "problem"

promptfile = "./cs336_alignment/prompts/r1_zero.prompt"
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
)
sampling_params.include_stop_str_in_output = True

with open(promptfile, "r", encoding="utf-8") as f:
    prompt_template = f.read()

llm = LLM(model="models/Qwen2.5-Math-1.5B")

inputs = []
prompts = []
answers = []
with open(datafile, "r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        inputs.append(j)
        answers.append(j[answer_key])
        prompt = prompt_template.format("question", question=j[question_key])
        prompts.append(prompt)


evaluate_vllm(llm, drgrpo_grader.r1_zero_reward_fn, prompts, answers, sampling_params)


