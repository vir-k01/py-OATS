"""Tests for py_oats.workflow.core — AmorphousStateMaker and ProductionMDMaker."""

from __future__ import annotations

import os
import gzip
import shutil
from pathlib import Path

import pytest
from pymatgen.core import Composition, Structure, Lattice
from pymatgen.io.lammps.data import LammpsData

from py_oats.workflow.core import (
    AmorphousStateMaker,
    ProductionMDMaker,
    AnalysisMaker,
    ProductionOnsagerMaker,
)
from py_oats.utils.workflow.helpers import (
    _species_string,
    _AMORPHOUS_STATE_DEFAULTS,
    _PRODUCTION_MD_DEFAULTS,
    TEMPLATE_DIR,
)


TEST_DATA = Path(__file__).parent / "test_data" / "flows"
REF_AMORPHOUS = TEST_DATA / "amorphous_state"
REF_PRODUCTION = TEST_DATA / "production_md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def li2s_structure():
    lattice = Lattice.cubic(13.5)
    species = ["Li"] * 68 + ["S"] * 34
    coords = [[i * 0.01, i * 0.02, i * 0.03] for i in range(102)]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


@pytest.fixture
def amorphous_maker():
    return AmorphousStateMaker(settings={
        **_AMORPHOUS_STATE_DEFAULTS.copy(),
        "t_npt": 0.5,
        "t_nvt": 0.5,
        "t_ramp": 0.5,
        "n_frames": 2,
    })


@pytest.fixture
def production_maker():
    return ProductionMDMaker(settings={
        **_PRODUCTION_MD_DEFAULTS.copy(),
        "temperature": 1000.0,
        "eq_steps": 500,
        "prod_steps": 1000,
        "log_interval": 100,
        "dump_interval": 100,
    })


# ---------------------------------------------------------------------------
# _species_string
# ---------------------------------------------------------------------------

def test_species_string_preserves_order(li2s_structure):
    result = _species_string(li2s_structure)
    assert result == "Li S"


def test_species_string_no_duplicates():
    lattice = Lattice.cubic(5.0)
    struct = Structure(lattice, ["O", "Li", "O", "Li"], [
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]
    ], coords_are_cartesian=True)
    assert _species_string(struct) == "O Li"


# ---------------------------------------------------------------------------
# Template files exist
# ---------------------------------------------------------------------------

def test_amorphous_template_exists():
    assert (TEMPLATE_DIR / "amorphous_state_job.in").is_file()


def test_production_template_exists():
    assert (TEMPLATE_DIR / "production_md_job.in").is_file()


# ---------------------------------------------------------------------------
# AmorphousStateMaker — instantiation and defaults
# ---------------------------------------------------------------------------

def test_amorphous_maker_defaults():
    maker = AmorphousStateMaker()
    assert maker.name == "amorphous_state_job"
    assert "potential_path" in maker.settings
    assert maker.settings["atom_style"] == "atomic"
    assert maker.settings["temperature"] == 300.0
    assert maker.settings["t_npt"] == 20.0


def test_amorphous_maker_settings_override():
    maker = AmorphousStateMaker(settings={
        **_AMORPHOUS_STATE_DEFAULTS.copy(),
        "temperature": 1500.0,
    })
    assert maker.settings["temperature"] == 1500.0
    assert maker.settings["t_npt"] == 20.0


def test_amorphous_maker_inputfile_path():
    maker = AmorphousStateMaker()
    assert Path(maker.inputfile).name == "amorphous_state_job.in"


def test_amorphous_maker_settings_are_independent():
    m1 = AmorphousStateMaker()
    m2 = AmorphousStateMaker()
    m1.settings["temperature"] = 9999.0
    assert m2.settings["temperature"] == 300.0


# ---------------------------------------------------------------------------
# ProductionMDMaker — instantiation and defaults
# ---------------------------------------------------------------------------

def test_production_maker_defaults():
    maker = ProductionMDMaker()
    assert maker.name == "production_md_job"
    assert maker.settings["atom_style"] == "atomic"
    assert maker.settings["pair_style"] == "grace"
    assert maker.settings["eq_steps"] == 50000
    assert maker.settings["prod_steps"] == 10000000


def test_production_maker_inputfile_path():
    maker = ProductionMDMaker()
    assert Path(maker.inputfile).name == "production_md_job.in"


def test_production_maker_settings_override():
    maker = ProductionMDMaker(settings={
        **_PRODUCTION_MD_DEFAULTS.copy(),
        "temperature": 2000.0,
        "prod_steps": 500,
    })
    assert maker.settings["temperature"] == 2000.0
    assert maker.settings["prod_steps"] == 500


# ---------------------------------------------------------------------------
# ProductionMDMaker — vel_dump_cmd construction
# ---------------------------------------------------------------------------

def test_vel_dump_cmd_two_species(production_maker, li2s_structure):
    production_maker.make(input_structure=li2s_structure)
    settings = production_maker.input_set_generator.settings.as_dict()
    vel_cmd = settings["vel_dump_cmd"]
    assert "$(step)" in vel_cmd
    assert "$(c_vcm[1][1])" in vel_cmd
    assert "$(c_vcm[2][3])" in vel_cmd
    assert "Li_vx" in vel_cmd
    assert "S_vz" in vel_cmd
    assert "species_vcm.dat" in vel_cmd


def test_vel_dump_cmd_three_species():
    lattice = Lattice.cubic(10.0)
    struct = Structure(lattice, ["Li", "Mn", "O"],
                       [[0, 0, 0], [2, 2, 2], [4, 4, 4]],
                       coords_are_cartesian=True)
    maker = ProductionMDMaker(settings=_PRODUCTION_MD_DEFAULTS.copy())
    maker.make(input_structure=struct)
    settings = maker.input_set_generator.settings.as_dict()
    vel_cmd = settings["vel_dump_cmd"]
    assert "$(c_vcm[3][1])" in vel_cmd
    assert "Mn_vy" in vel_cmd
    assert "O_vz" in vel_cmd


def test_species_string_injected(production_maker, li2s_structure):
    production_maker.make(input_structure=li2s_structure)
    settings = production_maker.input_set_generator.settings.as_dict()
    assert settings["species_string"] == "Li S"


# ---------------------------------------------------------------------------
# Input generation — validate against reference files
# ---------------------------------------------------------------------------

def test_amorphous_input_generates_valid_lammps(amorphous_maker, li2s_structure, tmp_path):
    amorphous_maker.input_set_generator.update_settings(
        {"species_string": _species_string(li2s_structure)}, validate_params=False
    )
    input_set = amorphous_maker.input_set_generator.get_input_set(data=li2s_structure)
    input_set.write_input(tmp_path)
    in_lammps = (tmp_path / "in.lammps").read_text()
    assert "read_data input.data" in in_lammps
    assert "pair_style" in in_lammps
    assert "Li S" in in_lammps
    assert "species_string" not in in_lammps
    assert (tmp_path / "input.data").is_file()


def test_production_input_generates_valid_lammps(production_maker, li2s_structure, tmp_path):
    production_maker.make(input_structure=li2s_structure)
    input_set = production_maker.input_set_generator.get_input_set(data=li2s_structure)
    input_set.write_input(tmp_path)
    in_lammps = (tmp_path / "in.lammps").read_text()
    assert "read_data input.data" in in_lammps
    assert "grace" in in_lammps
    assert "Li S" in in_lammps
    assert "species_vcm.dat" in in_lammps
    assert "production.dump" in in_lammps
    assert "production_final.data" in in_lammps


def test_amorphous_input_data_is_atomic_style(amorphous_maker, li2s_structure, tmp_path):
    amorphous_maker.input_set_generator.update_settings(
        {"species_string": _species_string(li2s_structure)}, validate_params=False
    )
    input_set = amorphous_maker.input_set_generator.get_input_set(data=li2s_structure)
    input_set.write_input(tmp_path)
    data_text = (tmp_path / "input.data").read_text()
    atoms_section = data_text.split("Atoms")[1].strip().split("\n")[1]
    fields = atoms_section.split()
    assert len(fields) == 5


def test_production_input_has_npt_volume_averaging(production_maker, li2s_structure, tmp_path):
    production_maker.make(input_structure=li2s_structure)
    input_set = production_maker.input_set_generator.get_input_set(data=li2s_structure)
    input_set.write_input(tmp_path)
    in_lammps = (tmp_path / "in.lammps").read_text()
    assert "f_avg_vol" in in_lammps or "avg_vol" in in_lammps
    assert "change_box" in in_lammps
    assert "vol_avg" in in_lammps


# ---------------------------------------------------------------------------
# Reference output parsing — amorphous state
# ---------------------------------------------------------------------------

def test_reference_amorphous_log_parses():
    from pymatgen.io.lammps.outputs import parse_lammps_log
    log_path = str(REF_AMORPHOUS / "log.lammps")
    runs = parse_lammps_log(log_path)
    assert len(runs) > 0
    for df in runs:
        assert "Step" in df.columns
        assert "Temp" in df.columns


def test_reference_amorphous_input_has_species():
    text = (REF_AMORPHOUS / "in.lammps").read_text()
    assert "Li S" in text
    assert "${species_string}" not in text


def test_reference_amorphous_input_consistent_thermo():
    text = (REF_AMORPHOUS / "in.lammps").read_text()
    for line in text.split("\n"):
        if line.strip().startswith("thermo_style"):
            assert "step time temp press pe ke etotal vol density" in line.lower().replace("  ", " ")


def test_reference_amorphous_data_file():
    text = (REF_AMORPHOUS / "input.data").read_text()
    assert "102  atoms" in text or "102 atoms" in text
    assert "2  atom types" in text or "2 atom types" in text


# ---------------------------------------------------------------------------
# Reference output parsing — production MD
# ---------------------------------------------------------------------------

def test_reference_production_log_parses():
    from pymatgen.io.lammps.outputs import parse_lammps_log
    log_path = str(REF_PRODUCTION / "log.lammps")
    runs = parse_lammps_log(log_path)
    assert len(runs) > 0


def test_reference_production_input_has_vel_dump():
    text = (REF_PRODUCTION / "in.lammps").read_text()
    assert "fix fvcm" in text
    assert "species_vcm.dat" in text
    assert "Li_vx" in text
    assert "S_vz" in text


def test_reference_production_input_has_write_data():
    text = (REF_PRODUCTION / "in.lammps").read_text()
    assert "write_data" in text


def test_reference_production_vcm_header():
    header = (REF_PRODUCTION / "species_vcm_header.dat").read_text()
    first_line = header.strip().split("\n")[0]
    assert "step" in first_line
    assert "Li_vx" in first_line
    assert "S_vz" in first_line


def test_reference_production_vcm_has_data():
    lines = (REF_PRODUCTION / "species_vcm_header.dat").read_text().strip().split("\n")
    assert len(lines) >= 2
    data_line = lines[1].split()
    assert len(data_line) == 7


# ---------------------------------------------------------------------------
# Reference output — task document parsing
# ---------------------------------------------------------------------------

def test_reference_amorphous_task_doc():
    from atomate2.lammps.schemas.task import LammpsTaskDocument
    doc = LammpsTaskDocument.from_directory(
        str(REF_AMORPHOUS), task_label="amorphous_state_job"
    )
    assert doc.state == "successful"
    assert doc.task_label == "amorphous_state_job"


def test_reference_production_task_doc():
    from atomate2.lammps.schemas.task import LammpsTaskDocument
    doc = LammpsTaskDocument.from_directory(
        str(REF_PRODUCTION), task_label="production_md_job"
    )
    assert doc.state == "successful"
    assert doc.task_label == "production_md_job"


# ---------------------------------------------------------------------------
# AnalysisMaker / ProductionOnsagerMaker — instantiation
# ---------------------------------------------------------------------------

def test_analysis_maker_defaults():
    from py_oats.analyzers.transport import TransportAnalyzer
    maker = AnalysisMaker()
    assert maker.name == "analysis_job"
    assert TransportAnalyzer in maker.analyzers


def test_production_onsager_maker_defaults():
    maker = ProductionOnsagerMaker()
    assert maker.name == "production_onsager_job"
    assert isinstance(maker.production_md_maker, ProductionMDMaker)
    assert isinstance(maker.analysis_maker, AnalysisMaker)
