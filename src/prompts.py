"""Prompt templates for the three methods.

direct : answer only, no explanation (zero-shot)
cot    : think step by step first (chain-of-thought)
grpo   : uses the SAME prompt as cot, so the comparison is fair
"""

TASK = ("Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use +, -, *, / and brackets. Each number must be used exactly once.")

DIRECT = (TASK + " Give only the equation inside <answer> </answer> tags, with no explanation. "
          "Example: <answer>(1 + 2) / 3</answer>")

COT = (TASK + " Think step by step inside <think> </think> tags. Keep your reasoning short. "
       "Then write the final equation inside <answer> </answer> tags, "
       "for example <answer>(1 + 2) / 3</answer>.")

TEMPLATES = {"direct": DIRECT, "cot": COT, "grpo": COT}

# Text we put at the START of the model's answer (prefill).
# The model continues from here, so it starts inside the <think> block.
PREFILL = {"direct": "", "cot": "<think>", "grpo": "<think>"}


def build_messages(numbers, target, method):
    """Chat messages for an Instruct model."""
    text = TEMPLATES[method].format(numbers=numbers, target=target)
    return [{"role": "user", "content": text}]


def build_prompt(tok, numbers, target, method):
    """Full prompt text: chat template + prefill."""
    prompt = tok.apply_chat_template(build_messages(numbers, target, method),
                                     tokenize=False, add_generation_prompt=True)
    return prompt + PREFILL[method]