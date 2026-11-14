"""
prompts.py - the prompts (questions) we give to the model
=========================================================

WHAT
    Builds the text the model receives for each puzzle, for our 3 methods.

THE 3 METHODS
    zero_shot : ask for the equation only, no reasoning, no examples
    cot       : zero-shot chain-of-thought: "think step by step" in <think>,
                then the equation in <answer>  (no solved examples, so it is
                closer to Kojima et al. 2022 than to the few-shot CoT of Wei et al. 2022)
    grpo      : EXACTLY the same prompt as cot. The only difference between
                cot and grpo is the training, so any gain comes from GRPO.

PREFILL
    For cot and grpo we write "<think>" ourselves at the start of the model's
    answer. Without it, small models ignored the tags and rambled until the
    length limit. The model's output then does NOT contain "<think>", so
    evaluate.py (and later the training rewards) add it back before checking.

The example "(1 + 2) / 3" in the prompts only shows the FORMAT; it is not a
solved puzzle, so the prompts stay zero-shot.
"""

TASK = ("Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use +, -, *, / and brackets. Each number must be used exactly once.")

ZERO_SHOT = (TASK + " Give only the equation inside <answer> </answer> tags, with no explanation. "
             "Example: <answer>(1 + 2) / 3</answer>")

COT = (TASK + " Think step by step inside <think> </think> tags. Keep your reasoning short. "
       "Then write the final equation inside <answer> </answer> tags, "
       "for example <answer>(1 + 2) / 3</answer>.")

TEMPLATES = {"zero_shot": ZERO_SHOT, "cot": COT, "grpo": COT}

# Text we put at the START of the model's answer (prefill).
PREFILL = {"zero_shot": "", "cot": "<think>", "grpo": "<think>"}


def build_messages(numbers, target, method):
    """Chat messages for an Instruct model (one user message)."""
    text = TEMPLATES[method].format(numbers=numbers, target=target)
    return [{"role": "user", "content": text}]


def build_prompt(tok, numbers, target, method):
    """Full prompt text = chat template (from the tokenizer) + prefill."""
    prompt = tok.apply_chat_template(build_messages(numbers, target, method),
                                     tokenize=False, add_generation_prompt=True)
    return prompt + PREFILL[method]