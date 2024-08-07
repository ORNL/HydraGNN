from enum import Enum
from typing import Annotated, List, Optional
import re
import json

import yaml
from pydantic import BaseModel, AfterValidator

def valid_split(s):
    if s.startswith("train"): return True
    if s.startswith("val"): return True
    if s.startswith("test"): return True
    if s.startswith("excl"): return True
    return False
SplitValue = Annotated[str, AfterValidator(valid_split)]

cat_re = re.compile(r"categorical\([1-9][0-9]*\)")
def valid_task_type(s):
    if s.startswith("num"): return True
    if s == "binary": return True
    if cat_re.fullmatch(s): return True
    return False
TypeValue = Annotated[str, AfterValidator(valid_task_type)]

class Task(BaseModel):
    name: str
    type: TypeValue
    description: str = ""

class DataDescriptor(BaseModel, extra="ignore"): # or "allow" to keep extra
    name:    str
    smiles:  str
    source:  str
    split:   Optional[str] = None
    authors: str = ""
    ref:     str = ""
    graph_tasks: List[Task] = []
    edge_tasks:  List[Task] = []
    node_tasks:  List[Task] = []

def main(argv):
    assert len(argv) == 2, f"Usage: {argv[0]} <descr.yaml>"
    with open(argv[1], "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    descr = DataDescriptor.model_validate(data)
    #print(descr.model_dump_json(indent=2, exclude_defaults=True))
    print( f"{descr.name}: {len(descr.graph_tasks)} tasks." )

if __name__=="__main__":
    import sys
    main(sys.argv)
