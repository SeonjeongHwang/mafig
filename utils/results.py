def get_last_state(example):
    last_round = max(example["trajectory"].keys())
    last_worker = example["trajectory"][last_round]["last_worker"]
    return example["trajectory"][last_round][last_worker]["state"]


def build_passage_result(example, include_status=False):
    result = {
        "id": example["id"],
        "source_id": example["source_id"],
        "level": example["level"],
        "source_text": example["input_data"]["source_text"],
        "constraints": example["constraints"],
        "passage": get_last_state(example),
    }

    if include_status:
        result["is_success"] = example["is_success"]
        result["is_terminated"] = example["is_terminated"]

    return result


def build_option_result(example, include_status=False):
    state = get_last_state(example)
    result = {
        "id": example["id"],
        "source_id": example["source_id"],
        "level": example["level"],
        "passage": example["input_data"]["passage"],
        "constraints": example["constraints"],
        "stem": state["stem"],
        "options": state["options"],
        "answer": state["answer"],
    }

    if include_status:
        result["is_success"] = example["is_success"]

    return result


def split_results_by_success(examples, build_result):
    all_results, success_results, fail_results = [], [], []

    for example in examples:
        result = build_result(example)
        all_results.append(result)

        if example["is_success"]:
            success_results.append(result)
        elif not example["is_terminated"]:
            fail_results.append(result)

    return all_results, success_results, fail_results


def select_one_result_per_source(results, drop_keys=()):
    selected = {
        result["id"].split("_sample")[0]: None
        for result in results
    }

    for result in results:
        source_id = result["id"].split("_sample")[0]
        if selected[source_id] is not None:
            continue
        if result["is_success"] is False:
            continue
        selected[source_id] = _copy_for_selected_source(result, source_id, drop_keys)

    for result in results:
        source_id = result["id"].split("_sample")[0]
        if selected[source_id] is not None:
            continue
        selected[source_id] = _copy_for_selected_source(result, source_id, drop_keys)

    return list(selected.values())


def _copy_for_selected_source(result, source_id, keys):
    selected = {
        key: value
        for key, value in result.items()
        if key not in keys
    }
    selected["id"] = source_id
    return selected
