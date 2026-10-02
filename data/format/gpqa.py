from datasets import load_dataset, Dataset
import random

PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
Problem: {problem}
Options: {options}
Answer:"""


def _generate_multiple_choice_answers_gpqa(data, rng: random.Random) -> tuple[str, str]:
    """Generate multiple choice string and correct answer letter."""
    answers = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    rng.shuffle(answers)

    options = ["A", "B", "C", "D"]
    options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

    multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
    correct_answer_letter = next(
        letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
    )
    return multiple_choice_string, correct_answer_letter


def _format_gpqa(example, idx: int, seed: int) -> dict:
    rng = random.Random(seed + idx)
    multiple_choice_string, answer = _generate_multiple_choice_answers_gpqa(example, rng)
    return {
        "kind": "gpqa",
        "dataset": "gpqa",
        "description": example["Question"],
        "problem": example["Question"],
        "prompt": PROMPT.format(problem=example["Question"], options=multiple_choice_string),
        "answer": answer,
    }


def load_gpqa(category: str = None, seed: int = 42) -> Dataset:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if category is not None: # Physics, Chemistry, Biology
        ds = ds.filter(lambda ex: ex["High-level domain"] == category)
    return ds.map(_format_gpqa, with_indices=True, fn_kwargs={"seed": seed}, remove_columns=ds.column_names)
