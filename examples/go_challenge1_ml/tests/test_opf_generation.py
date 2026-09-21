import pytest

pytest.importorskip("pandapower")

from go_challenge1.network_model import Bus, Generator, Load, NetworkScenario, SourceRef
from go_challenge1.opf_generation import generate_opf_sample


def _source():
    return SourceRef(source_file="x", source_line=1, raw_text="")


def test_opf_generation_returns_status():
    scenario = NetworkScenario(
        scenario_id="s1",
        network_name="n1",
        buses=[Bus(bus_id="1", bus_type=3, source=_source())],
        generators=[
            Generator(
                generator_id="1",
                bus_id="1",
                pg=10.0,
                pmin=0.0,
                pmax=100.0,
                qmin=-100.0,
                qmax=100.0,
                source=_source(),
            )
        ],
        loads=[Load(load_id="1", bus_id="1", pd=1.0, qd=0.1, source=_source())],
    )
    result = generate_opf_sample(scenario)
    assert "success" in result
    assert "termination_status" in result
