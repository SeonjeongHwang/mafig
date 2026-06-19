import json


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data, indent=4):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def write_marker(path, content="-"):
    with open(path, "w") as f:
        f.write(content)


def restore_trajectory_round_keys(example):
    restored = {key: value for key, value in example.items() if key != "trajectory"}
    restored["trajectory"] = {
        int(round_id): trajectory
        for round_id, trajectory in example["trajectory"].items()
    }
    return restored


def restore_examples_trajectory(examples):
    return [restore_trajectory_round_keys(example) for example in examples]
