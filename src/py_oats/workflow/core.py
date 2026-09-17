from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, Response, job
from pymatgen.core import Composition, Structure

from atomate2.lammps.jobs.core import CustomLammpsMaker
from atomate2.lammps.schemas.task import StoreTrajectoryOption
from py_oats.analyzers.base import BaseAnalyzer
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.structure_generator.generator import get_amorphous_structure
from py_oats.utils.workflow.helpers import (
    TEMPLATE_DIR,
    _AMORPHOUS_STATE_DEFAULTS,
    _ANALYZER_TO_SCHEMA,
    _PRODUCTION_MD_DEFAULTS,
    _build_species_settings,
    _find_trajectory_file,
    _species_string,
)


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

    When called directly with a resolved Structure, species_string and
    vel_dump_cmd are computed at build time. When chained via
    ProductionOnsagerMaker with an OutputReference, use
    ``run_production_md`` instead — it resolves the structure at runtime.
    """

    name: str = "production_md_job"
    inputfile: str | Path = field(
        default_factory=lambda: TEMPLATE_DIR / "production_md_job.in"
    )
    settings: dict = field(default_factory=lambda: _PRODUCTION_MD_DEFAULTS.copy())
    task_document_kwargs: dict = field(default_factory=lambda: {"store_trajectory": StoreTrajectoryOption.PARTIAL})

    def make(self, input_structure: Structure | None = None, **kwargs):
        if input_structure is not None and isinstance(input_structure, Structure):
            self.input_set_generator.update_settings(
                _build_species_settings(input_structure),
                validate_params=False,
            )
        return super().make(input_structure=input_structure, **kwargs)


@job
def run_production_md(
    input_structure: Structure,
    production_md_maker: ProductionMDMaker,
) -> Response:
    """Resolve species at runtime and replace with the real LAMMPS job.

    Use this instead of ``ProductionMDMaker.make()`` when
    ``input_structure`` is an ``OutputReference`` (e.g. from a prior job).
    """
    production_md_maker.input_set_generator.update_settings(
        _build_species_settings(input_structure),
        validate_params=False,
    )
    lammps_job = production_md_maker.make(input_structure=input_structure)
    return Response(replace=lammps_job)


@dataclass
class AnalysisMaker(Maker):

    name: str = "analysis_job"
    analyzers: list[type[BaseAnalyzer]] = field(
        default_factory=lambda: [TransportAnalyzer]
    )
    temperature: float = 300.0
    time_step: float = 0.001
    dump_interval: int = 1000

    @job
    def make(self, run_dir: str):
        trajectory_file = _find_trajectory_file(run_dir)
        data = TrajectoryData.read(
            trajectory_file,
            temperature=self.temperature,
            time_step=self.time_step,
            step_skip=self.dump_interval,
        )
        docs = []
        for analyzer_cls in self.analyzers:
            analyzer_instance = analyzer_cls(data)
            analyzer_instance.analyze()
            schema_cls = _ANALYZER_TO_SCHEMA.get(analyzer_cls)
            if schema_cls is not None and hasattr(schema_cls, "from_analyzer"):
                doc = schema_cls.from_analyzer(analyzer_instance)
            else:
                doc = {"analyzer": analyzer_cls.__name__, "species": list(data.unique_species)}
            docs.append(doc)
        return docs


@dataclass
class ProductionOnsagerMaker(Maker):

    name: str = "production_onsager_job"
    production_md_maker: ProductionMDMaker = field(
        default_factory=ProductionMDMaker
    )
    analysis_maker: AnalysisMaker = field(default_factory=AnalysisMaker)

    def __post_init__(self):
        md_settings = self.production_md_maker.settings
        self.analysis_maker.temperature = md_settings.get(
            "temperature", self.analysis_maker.temperature
        )
        self.analysis_maker.time_step = md_settings.get(
            "time_step", self.analysis_maker.time_step
        )
        self.analysis_maker.dump_interval = md_settings.get(
            "dump_interval", self.analysis_maker.dump_interval
        )

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

        production_md_job = run_production_md(
            input_structure=structure,
            production_md_maker=self.production_md_maker,
        )
        analysis_job = self.analysis_maker.make(
            run_dir=production_md_job.output.dir_name
        )

        jobs.append(production_md_job)
        jobs.append(analysis_job)

        return Flow(jobs, name=self.name)
