import pytest

pytest.importorskip("pandapower")

from go_challenge1.network_model import Bus, Generator, NetworkScenario, SourceRef
from go_challenge1.pandapower_export import network_to_pandapower


def _source():
    return SourceRef(source_file="x", source_line=1, raw_text="")


def test_network_to_pandapower_basic():
    scenario = NetworkScenario(
        scenario_id="s1",
        network_name="n1",
        buses=[Bus(bus_id="1", source=_source())],
        generators=[Generator(generator_id="1", bus_id="1", source=_source())],
    )
    net, bus_map = network_to_pandapower(scenario)
    assert len(net.bus) == 1
    assert "1" in bus_map


def test_id_maps_keep_original_and_zero_based_indices():
    scenario = NetworkScenario(
        scenario_id="s1",
        network_name="n1",
        buses=[Bus(bus_id="101", source=_source())],
        generators=[Generator(generator_id="A", bus_id="101", source=_source())],
    )
    maps = scenario.id_maps()
    assert scenario.buses[0].bus_id == "101"
    assert maps["bus"]["101"] == 0
    assert maps["generator"]["101:A"] == 0
