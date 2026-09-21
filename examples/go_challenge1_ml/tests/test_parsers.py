from pathlib import Path

import pytest

from go_challenge1.discovery import ScenarioFiles
from go_challenge1.parser_con import parse_con_file
from go_challenge1.parser_inl import parse_inl_file
from go_challenge1.parser_raw import parse_raw_file
from go_challenge1.parser_rop import parse_rop_file
from go_challenge1.parser_utils import parse_scenario_bundle


def test_parse_raw_minimal(tmp_path: Path):
    raw = tmp_path / "case.raw"
    raw.write_text(
        "0, 100.0, 0, 0, 0, 60.0 /\n"
        "1,'BUS1',138,3,1,1,1,1.00,0.0,1.1,0.9\n"
        "0 / END OF BUS DATA, BEGIN LOAD DATA\n"
        "1,'1',1,1,1,50,30\n"
        "0 / END OF LOAD DATA, BEGIN GENERATOR DATA\n"
        "1,'1',40,5,50,-40,1.0,0,100,1,100,1,0,0,1,100,10\n"
        "0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\n"
        "1,1,'1',0.0,0.01,0.0,100,100,100,0,0,0,0,1\n"
        "0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA\n"
        "0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\n",
        encoding="utf-8",
    )
    scenario = parse_raw_file(raw)
    assert scenario.base_mva == 100.0
    assert len(scenario.buses) == 1
    assert len(scenario.loads) == 1
    assert len(scenario.generators) == 1


def test_parse_rop_minimal(tmp_path: Path):
    rop = tmp_path / "case.rop"
    rop.write_text("1,'1',0.0,1.0,2.0\n", encoding="utf-8")
    curves = parse_rop_file(rop)
    assert len(curves) == 1
    assert curves[0].bus_id == "1"


def test_parse_inl_minimal(tmp_path: Path):
    inl = tmp_path / "case.inl"
    inl.write_text("1,'1',0.75\n", encoding="utf-8")
    factors = parse_inl_file(inl)
    assert factors[("1", "1")] == 0.75


def test_parse_con_minimal(tmp_path: Path):
    con = tmp_path / "case.con"
    con.write_text(
        "CONTINGENCY C1\n"
        "OPEN BRANCH FROM BUS 1 TO BUS 2 CKT '1'\n"
        "END\n",
        encoding="utf-8",
    )
    contingencies = parse_con_file(con)
    assert len(contingencies) == 1


def test_duplicate_bus_detection(tmp_path: Path):
    raw = tmp_path / "dup.raw"
    raw.write_text(
        "0, 100.0, 0, 0, 0, 60.0 /\n"
        "1,'BUS1',138,3,1,1,1,1.00,0.0,1.1,0.9\n"
        "1,'BUS1_DUP',138,1,1,1,1,1.00,0.0,1.1,0.9\n"
        "0 / END OF BUS DATA, BEGIN LOAD DATA\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate bus identifier"):
        parse_raw_file(raw)


def test_unknown_generator_in_rop_detection(tmp_path: Path):
    root = tmp_path / "scenario_a"
    root.mkdir(parents=True)

    raw = root / "case.raw"
    raw.write_text(
        "0, 100.0, 0, 0, 0, 60.0 /\n"
        "1,'BUS1',138,3,1,1,1,1.00,0.0,1.1,0.9\n"
        "0 / END OF BUS DATA, BEGIN GENERATOR DATA\n"
        "1,'1',10,0,50,-50,1.0,0,100,1,100,1,0,0,1,100,0\n"
        "0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\n",
        encoding="utf-8",
    )

    rop = root / "case.rop"
    rop.write_text("999,'2',0.0,1.0,2.0\n", encoding="utf-8")

    files = ScenarioFiles(
        scenario_id="scenario_a",
        network_name="case",
        path=root,
        raw=raw,
        rop=rop,
        inl=None,
        con=None,
    )

    with pytest.raises(ValueError, match="Unknown generator referenced in ROP file"):
        parse_scenario_bundle(files)
