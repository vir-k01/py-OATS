from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, job
from pymatgen.core import Composition, Structure

from atomate2.lammps.jobs.core import CustomLammpsMaker
from py_oats.analyzers.base import BaseAnalyzer
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.structure_generator.generator import get_amorphous_structure
from atomate2.lammps.schemas.task import StoreTrajectoryOption


TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

_AMORPHOUS_STATE_DEFAULTS: dict = {
    "atom_style": "atomic",
    "potential_path": str(Path.home() / ".cache/grace/GRACE-FS-OAM"),
    "temperature": 300.0,
    "pressure": 0.0,
    "time_step": 0.001,
    "t_damp": 0.1,
    "p_damp": 1.0,
    "t_npt": 20.0,
    "t_nvt": 20.0,
    "t_ramp": 20.0,
    "ramp_temperature": 5000.0,
    "thermo_interval": 1000,
    "dump_interval": 1000,
    "density_threshold": 0.30,
    "n_frames": 10,
    "e_tol": 1.0e-10,
    "f_tol": 1.0e-10,
    "max_iter": 20000,
    "max_eval": 200000,
}


_PRODUCTION_MD_DEFAULTS: dict = {
    "atom_style": "atomic",
    "pair_style": "grace",
    "potential_path": str(Path.home() / ".cache/grace/GRACE-FS-OAM"),
    "temperature": 300.0,
    "time_step": 0.001,
    "log_interval": 100,
    "dump_interval": 1000,
    "eq_steps": 50000,
    "prod_steps": 10000000,
    "seed": 12345,
}


def _species_string(structure: Structure) -> str:
    """Return a space-separated species string preserving element order."""
    seen: set[str] = set()
    species: list[str] = []
    for site in structure:
        sym = site.specie.symbol
        if sym not in seen:
            seen.add(sym)
            species.append(sym)
    return " ".join(species)


@dataclass
class AmorphousStateMaker(CustomLammpsMaker):
    """
    A ``CustomLammpsMaker`` that generates an amorphous structure from a
    composition and uses LAMMPS with a fast GRACE-FS potential to equilibrate it
    and a bigger GRACE-3L potential to relax it a local minimum (keeping a cubic cell). 
    This is akin to generating the structures to use for ``amorphous limit'' calculations.
    The output structures of this job can be used as inputs for other MD jobs as well.
    """

    name: str = "amorphous_state_job"
    inputfile: str | Path = field(
        default_factory=lambda: TEMPLATE_DIR / "amorphous_state_job.in"
    )
    settings: dict = field(default_factory=lambda: _AMORPHOUS_STATE_DEFAULTS.copy())

    def make(self, composition: Composition, **kwargs):
        amorphous_structure = get_amorphous_structure(composition)
        species = _species_string(amorphous_structure)
        self.input_set_generator.update_settings(
            {"species_string": species}, validate_params=False
        )
        return super().make(input_structure=amorphous_structure, **kwargs)


@dataclass
class ProductionMDMaker(CustomLammpsMaker):
    """
    A ``CustomLammpsMaker`` that generates a production MD job.
    This job is used to calculate the transport properties for a given system.
    The input structure for this job should be an equilibrated structure from
    a previous job, such as an NPT or NVT job. A "guess" structure from
    `get_amorphous_structure` can also be used. A well-suited MLIP should be
    provided for this job. In the absence of an MLIP, the GRACE-FS potential will
    be used as a default.
    """

    name: str = "production_md_job"
    inputfile: str | Path = field(
        default_factory=lambda: TEMPLATE_DIR / "production_md_job.in"
    )
    settings: dict = field(default_factory=lambda: _PRODUCTION_MD_DEFAULTS.copy())
    task_document_kwargs: dict = field(default_factory=lambda: {"store_trajectory": StoreTrajectoryOption.PARTIAL})

    def make(self, input_structure: Structure | None = None, **kwargs):
        if input_structure is not None:
            species = _species_string(input_structure)
            elements = species.split()
            fmt_parts = ["$(step)"]
            for i in range(1, len(elements) + 1):
                fmt_parts += [f"$(c_vcm[{i}][{d}])" for d in (1, 2, 3)]
            title_parts = ["step"]
            for e in elements:
                title_parts += [f"{e}_vx", f"{e}_vy", f"{e}_vz"]
            vel_dump_cmd = (
                f'"{" ".join(fmt_parts)}" '
                f'file species_vcm.dat screen no '
                f'title "# {" ".join(title_parts)}"'
            )
            self.input_set_generator.update_settings(
                {"species_string": species, "vel_dump_cmd": vel_dump_cmd},
                validate_params=False,
            )
        return super().make(input_structure=input_structure, **kwargs)


@dataclass
class AnalysisMaker(Maker):

    name: str = "analysis_job"
    analyzers: list[type[BaseAnalyzer]] = field(
        default_factory=lambda: [TransportAnalyzer]
    )

    @job
    def make(self, run_dir: str, **kwargs):
        trajectory_file = os.path.join(run_dir, "production.dump")
        data = TrajectoryData.from_file(trajectory_file)
        docs = []
        for analyzer_cls in self.analyzers:
            analyzer_instance = analyzer_cls(data)
            analyzer_instance.analyze(**kwargs)
            doc = analyzer_instance.to_doc()
            docs.append(doc)
        return docs


@dataclass
class ProductionOnsagerMaker(Maker):

    name: str = "production_onsager_job"
    production_md_maker: ProductionMDMaker = field(
        default_factory=ProductionMDMaker
    )
    analysis_maker: AnalysisMaker = field(default_factory=AnalysisMaker)

    def make(self, structure_or_composition: Structure | Composition, **kwargs):
        jobs = []
        if isinstance(structure_or_composition, Composition):
            structure_job = AmorphousStateMaker().make(
                composition=structure_or_composition
            )
            jobs.append(structure_job)
            structure = structure_job.output.structure
        else:
            structure = structure_or_composition

        production_md_job = self.production_md_maker.make(
            input_structure=structure, **kwargs
        )
        analysis_job = self.analysis_maker.make(
            run_dir=production_md_job.output.dir_name
        )

        jobs.append(production_md_job)
        jobs.append(analysis_job)

        return Flow(jobs, name=self.name)
