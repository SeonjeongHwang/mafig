import numpy as np


OPTION_LABELS = ["A", "B", "C", "D"]


def build_passage_examples(source_data_list, level_constraints, already_success_ids):
    already_success_ids = set(already_success_ids)
    examples = []
    skipped_ids = []

    for data in source_data_list:
        for level, constraints in level_constraints.items():
            example_id = f"{data['id']}_level{level}"
            if example_id in already_success_ids:
                skipped_ids.append(example_id)
                continue

            examples.append({
                "id": example_id,
                "source_id": data["id"],
                "level": level,
                "input_data": {"source_text": data["document"]},
                "constraints": constraints["passage"],
                "trajectory": dict(),
                "planner_history": [],
                "is_success": False,
                "is_terminated": False,
            })

    return examples, skipped_ids


def normalize_passage_data_list(passage_data):
    if not isinstance(passage_data, dict):
        return passage_data, False

    passage_data_list = []
    for example_id, data in passage_data.items():
        normalized = dict(data)
        normalized["id"] = example_id
        normalized["source_id"] = example_id.split("_level")[0]
        normalized["level"] = example_id.split("_level")[-1]
        passage_data_list.append(normalized)

    return passage_data_list, True


def build_human_passage_option_examples(passage_data_list, level_constraints, success_ids):
    success_ids = set(success_ids)
    examples = []
    skipped_ids = []

    for data in passage_data_list:
        passage_id = data["id"]
        vocab_level = data["vocab_level"]

        for level, constraints in level_constraints.items():
            example_id = f"{passage_id}_level{level}"
            if example_id in success_ids:
                skipped_ids.append(example_id)
                continue

            examples.append(_build_option_example(
                example_id=example_id,
                source_id=passage_id,
                level=level,
                passage=data["passage"],
                vocab_level=vocab_level,
                option_constraints=constraints["option"],
            ))

    return examples, skipped_ids


def build_generated_passage_option_examples(passage_data_list, level_constraints, success_ids):
    success_ids = set(success_ids)
    examples = []
    skipped_ids = []

    for data in passage_data_list:
        example_id = data["id"]
        if example_id in success_ids:
            skipped_ids.append(example_id)
            continue

        level = data["level"]
        examples.append(_build_option_example(
            example_id=example_id,
            source_id=data["source_id"],
            level=level,
            passage=data["passage"],
            vocab_level=level_constraints[level]["passage"]["vocab_level"],
            option_constraints=level_constraints[level]["option"],
        ))

    return examples, skipped_ids


def _build_option_example(example_id, source_id, level, passage, vocab_level, option_constraints):
    shuffled_constraints = option_constraints[:]
    np.random.shuffle(shuffled_constraints)

    return {
        "id": example_id,
        "source_id": source_id,
        "level": level,
        "input_data": {"passage": passage},
        "constraints": {
            "vocab_level": vocab_level,
            "options": dict(zip(OPTION_LABELS, shuffled_constraints)),
        },
        "trajectory": dict(),
        "planner_history": [],
        "success_details": dict((oidx, False) for oidx in OPTION_LABELS),
        "is_success": False,
        "is_terminated": False,
    }
