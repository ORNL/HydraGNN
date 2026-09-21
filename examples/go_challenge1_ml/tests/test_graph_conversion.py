import torch

from go_challenge1.graph_conversion import scenario_solution_to_heterodata
from go_challenge1.network_model import (
    Branch,
    Bus,
    FixedShunt,
    Generator,
    Load,
    NetworkScenario,
    SourceRef,
)


def _source():
    return SourceRef(source_file="x", source_line=1, raw_text="")


def test_graph_conversion_shapes():
    scenario = NetworkScenario(
        scenario_id="s1",
        network_name="n1",
        buses=[Bus(bus_id="1", source=_source()), Bus(bus_id="2", source=_source())],
        generators=[Generator(generator_id="1", bus_id="1", source=_source())],
        loads=[Load(load_id="1", bus_id="2", pd=5.0, qd=1.0, source=_source())],
        fixed_shunts=[FixedShunt(shunt_id="1", bus_id="2", source=_source())],
        branches=[Branch(branch_id="1:2:1", from_bus_id="1", to_bus_id="2", source=_source())],
    )
    solution = {
        "solver_name": "test",
        "termination_status": "ok",
        "bus_solution": {"0": {"vm_pu": 1.0, "va_degree": 0.0, "p_mw": 0.0, "q_mvar": 0.0}},
        "generator_solution": {"0": {"p_mw": 1.0, "q_mvar": 0.0, "vm_pu": 1.0}},
    }
    data = scenario_solution_to_heterodata(scenario, solution, task_name="pf")
    assert data["bus"].x_static.shape[0] == 2
    assert data["generator"].x_static.shape[0] == 1
    assert data["bus", "line", "bus"].edge_index.shape[1] == 1
    assert int(data.task_id.item()) == 0
    assert torch.all(data["generator"].pg_is_observed == 1.0)
