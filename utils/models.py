from enum import Enum
from typing import Annotated, List, Optional
import re
import json

import yaml
from pydantic import BaseModel, AfterValidator


def valid_split(s):
    if s.startswith("train"):
        return s
    if s.startswith("val"):
        return s
    if s.startswith("test"):
        return s
    if s.startswith("excl"):
        return s
    raise ValueError(f"Split name: {s}")


SplitValue = Annotated[str, AfterValidator(valid_split)]

cat_re = re.compile(r"categorical\(([1-9][0-9]*)\)")


def number_categories(s):
    if s.startswith("num"):
        return 0
    if s == "regression":
        return 0
    if s == "binary":
        return 2
    match = cat_re.fullmatch(s)
    if match:
        return int(match[1])
    raise ValueError(f"Invalid task type: {s}")


def valid_task_type(s):
    if s.startswith("num"):
        return s
    if s == "regression":
        return s
    if s == "binary":
        return s
    if cat_re.fullmatch(s):
        return s
    raise ValueError(f"Invalid task type: {s}")


# TODO: fix this validator to check all types match a pattern
TypeValue = Annotated[str, AfterValidator(valid_task_type)]


class Task(BaseModel):
    name: str  # column name for task
    type: TypeValue  # type of task
    description: str = ""


class DataDescriptor(BaseModel, extra="ignore"):  # or "allow" to keep extra
    name: str
    smiles: str
    source: str
    split: Optional[str] = None  # column name for split
    authors: str = ""  # list of authors
    ref: str = ""
    graph_tasks: List[Task] = []  # list of prediction tasks for a graph
    edge_tasks: List[Task] = []
    node_tasks: List[Task] = []


def main(argv):
    assert len(argv) >= 2, f"Usage: {argv[0]} <descr.yaml> ..."
    for fname in argv[1:]:
        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        try:
            descr = DataDescriptor.model_validate(data)
        except Exception as e:
            print(f"{fname}: error loading yaml")
            short = " ".join(str(e).split("\n")[:3])
            short = short.split("[")[0]
            print("    " + short)
            continue
        # print(descr.model_dump_json(indent=2, exclude_defaults=True))

        print(f"{fname}:")
        print(f'    name: "{descr.name}"')
        print(f'    ref: "{descr.ref}"')
        print(f'    source: "{descr.source}"')
        print(f'    authors: "{descr.authors}"')
        print(f"    node tasks: {len(descr.node_tasks)}")
        print(f"    edge tasks: {len(descr.edge_tasks)}")
        print(f"    graph tasks: {len(descr.graph_tasks)}")
        print()


if __name__ == "__main__":
    import sys

    main(sys.argv)
