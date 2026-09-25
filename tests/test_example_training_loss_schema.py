import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
INTERATOMIC_TERMS = {"energy", "energy_per_atom", "forces"}
OPF_OPERATORS = {
    "voltage_limits": "bounded",
    "angle_limits": "edge_difference_bounded",
    "thermal_limits": "ac_thermal_limit",
}


def _example_configs():
    for path in EXAMPLES.rglob("*.json"):
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "NeuralNetwork" in config:
            yield path, config


def _loss_configs(provider):
    for path, config in _example_configs():
        training = config["NeuralNetwork"].get("Training", {})
        loss = training.get("loss")
        if isinstance(loss, dict) and loss.get("provider") == provider:
            yield path, config, loss


def test_examples_do_not_use_removed_training_loss_structure():
    for path, config in _example_configs():
        training = config["NeuralNetwork"].get("Training", {})
        assert "DomainLoss" not in training, path
        loss = training.get("loss", {})
        assert "terms" not in loss, path

    for path in EXAMPLES.rglob("*.py"):
        source = path.read_text(errors="ignore")
        assert '["loss"]["terms"]' not in source, path


@pytest.mark.parametrize(
    "path,config,loss", list(_loss_configs("interatomic_potential"))
)
def test_interatomic_examples_use_declarative_conservative_loss(path, config, loss):
    assert loss.get("enabled") is True, path
    supervised = loss["supervised"]
    assert supervised.get("default_metric"), path
    terms = supervised["terms"]
    assert isinstance(terms, list) and terms, path
    by_variable = {term["variable"]: term for term in terms}
    assert len(by_variable) == len(terms), path
    assert set(by_variable) <= INTERATOMIC_TERMS, path
    assert all(float(term["weight"]) > 0 for term in terms), path
    assert loss.get("constraints") == [], path
    assert loss.get("constraint_optimizer") == {"type": "fixed_penalty"}, path

    if "energy" in by_variable:
        assert by_variable["energy"].get("normalization") == "per_structure", path
    if "energy_per_atom" in by_variable:
        assert by_variable["energy_per_atom"].get("normalization") == "per_atom", path
    if "forces" in by_variable:
        assert by_variable["forces"].get("prediction", {}).get("operator") == (
            "negative_gradient"
        ), path


@pytest.mark.parametrize("path,config,loss", list(_loss_configs("optimal_power_flow")))
def test_opf_examples_use_named_constraint_schema(path, config, loss):
    assert "supervised" in loss, path
    assert isinstance(loss.get("constraints"), list), path
    assert loss.get("constraint_optimizer", {}).get("type") in {
        "fixed_penalty",
        "augmented_lagrangian",
    }, path

    constraints = {item["name"]: item for item in loss["constraints"]}
    assert len(constraints) == len(loss["constraints"]), path
    assert constraints and set(constraints) <= set(OPF_OPERATORS), path
    for name, constraint in constraints.items():
        operator = OPF_OPERATORS[name]
        assert constraint.get("operator") == operator, path

    variables = config["Variables"]
    outputs = {
        component
        for output in variables["outputs"]
        for component in output.get("components", [output["name"]])
    }
    inputs = {
        component
        for item in variables["inputs"]
        for component in item.get("components", [item["name"]])
    }
    voltage = constraints.get("voltage_limits")
    if voltage:
        assert voltage["value"] in outputs, path
        assert voltage["lower"] in inputs and voltage["upper"] in inputs, path

    edge_attributes = variables["edge_attributes"]
    for name in ("angle_limits", "thermal_limits"):
        constraint = constraints.get(name)
        if not constraint:
            continue
        for relation in constraint["relations"]:
            assert relation in edge_attributes, path
    angle = constraints.get("angle_limits")
    if angle:
        for relation in angle["relations"]:
            assert angle["lower"] in edge_attributes[relation], path
            assert angle["upper"] in edge_attributes[relation], path
