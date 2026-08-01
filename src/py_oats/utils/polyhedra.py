"""Extract polyhedral environments from Materials Project ground state structures."""

from __future__ import annotations

from itertools import combinations

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Molecule, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Geometry types recognised as network-forming polyhedra (from robocrystallographer).
_CONNECTED_GEOMETRIES: frozenset[str] = frozenset({
    "tetrahedral",
    "octahedral",
    "trigonal pyramidal",
    "square pyramidal",
    "trigonal bipyramidal",
    "pentagonal pyramidal",
    "hexagonal pyramidal",
    "pentagonal bipyramidal",
    "hexagonal bipyramidal",
    "cuboctahedral",
})

# Minimum order-parameter score for a geometry to be accepted.
_MINIMUM_GEOMETRY_OP: float = 0.4


def _chemsys_list(comp: Composition) -> list[str]:
    """
    Return the chemsys strings to query for a given composition.

    - Unary  : just the element
    - Binary : both unaries + the binary
    - Ternary+: all unaries + all pairwise binaries (no higher-order systems)
    """
    elements = sorted(str(el) for el in comp.elements)
    n = len(elements)

    if n == 1:
        return elements[:]

    chemsys = elements[:]

    if n == 2:
        chemsys.append("-".join(elements))
    else:
        for pair in combinations(elements, 2):
            chemsys.append("-".join(pair))

    return chemsys


def _classify_sites(structure: Structure) -> list[str | None]:
    """
    Classify every site's local geometry using CrystalNNFingerprint order parameters.

    Returns a list (one entry per site) of the best-matching geometry name from
    ``_CONNECTED_GEOMETRIES``, or ``None`` if the site does not qualify.

    Replicates robocrystallographer's ``SiteAnalyzer.get_site_geometry``:
      - Feature labels have the form ``"<geometry> CN_<n>"``; the last token is
        stripped to get the geometry name.
      - Labels containing ``"wt"`` are weight features and are skipped.
      - A geometry is accepted only when its order-parameter score >=
        ``_MINIMUM_GEOMETRY_OP`` *and* it appears in ``_CONNECTED_GEOMETRIES``.
    """
    from matminer.featurizers.site import CrystalNNFingerprint

    fp = CrystalNNFingerprint.from_preset("ops")
    labels = fp.feature_labels()

    geometries: list[str | None] = []
    for idx in range(len(structure)):
        try:
            features = fp.featurize(structure, idx)
        except Exception:
            geometries.append(None)
            continue

        best_geom: str | None = None
        best_score = 0.0
        for label, score in zip(labels, features):
            # Strip trailing "CN_N" token to get the geometry name.
            geom = " ".join(label.split()[:-1])
            if geom == "wt":
                continue
            if score > best_score:
                best_score = score
                best_geom = geom

        if best_score >= _MINIMUM_GEOMETRY_OP and best_geom in _CONNECTED_GEOMETRIES:
            geometries.append(best_geom)
        else:
            geometries.append(None)

    return geometries


def _polyhedra_from_structure(
    structure: Structure,
    nn_finder: CrystalNN,
    site_geometries: list[str | None],
) -> list[Molecule]:
    """
    Return one Molecule per site that passes the robocrystallographer polyhedra-former
    criteria:

    1. The site's own geometry must be in ``_CONNECTED_GEOMETRIES``.
    2. At least one **next-nearest neighbour** (NNN — a neighbour of a neighbour)
       must also have a connected geometry.  This mirrors robocrys's connectivity
       check: it ensures the polyhedron is part of a network rather than isolated.

    Gate 2 uses NNN rather than NN because the ligand species (e.g. O in SiO2)
    typically do not have a polyhedral geometry themselves; the connectivity is
    established via the next shell — other cation polyhedra sharing those ligands.
    """
    # Pre-compute nearest-neighbour lists for every site so NNN lookups reuse
    # the same CrystalNN calls without redundant computation.
    all_nn: dict[int, list[dict]] = {}
    for idx in range(len(structure)):
        try:
            all_nn[idx] = nn_finder.get_nn_info(structure, idx)
        except Exception:
            all_nn[idx] = []

    polyhedra = []
    for idx in range(len(structure)):
        # Gate 1: site must have a recognised polyhedral geometry.
        if site_geometries[idx] is None:
            continue

        nn_info = all_nn[idx]
        if not nn_info:
            continue

        # Gate 2: at least one NNN (neighbour-of-neighbour, excluding self) must
        # have a connected geometry.  This is the robocrys "connectivity" check.
        nnn_connected = False
        for nn in nn_info:
            for nnn in all_nn.get(nn["site_index"], []):
                if nnn["site_index"] != idx and site_geometries[nnn["site_index"]] is not None:
                    nnn_connected = True
                    break
            if nnn_connected:
                break

        if not nnn_connected:
            continue

        center = structure[idx]
        species = [center.specie]
        coords: list[list[float]] = [[0.0, 0.0, 0.0]]

        for nn in nn_info:
            nn_site = nn["site"]
            species.append(nn_site.specie)
            coords.append((nn_site.coords - center.coords).tolist())

        polyhedra.append(Molecule(species, coords))

    return polyhedra


_MAX_POLYHEDRA_ATOMS: int = 9


def get_polyhedra_from_mp(
    comp: Composition | str,
    mp_api_key: str | None = None,
    max_atoms: int = _MAX_POLYHEDRA_ATOMS,
) -> list[Molecule]:
    """
    Extract polyhedral environments from MP ground state structures.

    The subsystems queried depend on the number of elements in *comp*:

    - **Unary**   – only the element itself
    - **Binary**  – both unaries + the binary
    - **Ternary+** – all unaries + all pairwise binaries (no higher-order systems)

    Within each queried chemsys only convex-hull-stable (``is_stable=True``)
    structures are considered.  Local coordination environments are extracted
    with CrystalNN and filtered with the robocrystallographer approach:

    - **Gate 1** – site geometry must score >= 0.4 on a known polyhedral OP
      (``CrystalNNFingerprint`` from matminer).
    - **Gate 2** – at least one next-nearest neighbour must also have a connected
      geometry (polyhedra are network-forming, not isolated).

    Results are deduplicated so that at most one polyhedron per unique reduced
    formula is returned.

    Parameters
    ----------
    comp : Composition | str
        Composition whose element set defines the search space.
    mp_api_key : str | None
        MP API key.  Falls back to the ``MP_API_KEY`` environment variable if
        *None*.
    max_atoms : int
        Discard polyhedra with more than this many atoms (center + ligands).
        Default 9 keeps up to 8-coordinate environments (e.g. octahedra = 7)
        and filters out large metallic coordination shells (e.g. cuboctahedra
        = 13) that cause packmol convergence failures.

    Returns
    -------
    list[Molecule]
        One ``Molecule`` per unique polyhedral composition.  The central atom
        sits at the origin; neighbours are given as relative Cartesian
        coordinates in Å.
    """
    from mp_api.client import MPRester

    if isinstance(comp, (str, dict)):
        comp = Composition(comp)

    chemsys = _chemsys_list(comp)

    with MPRester(api_key=mp_api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            is_stable=True,
            fields=["structure"],
        )
    # Convert to conventional standard cells (mirrors robocrystallographer).
    # Primitive cells can be too small for the NNN connectivity check: in a
    # 2-atom rocksalt primitive cell every NNN of Mg (reached via O) has the
    # same site_index as Mg itself, so gate 2 incorrectly rejects all sites.
    # The conventional cell guarantees multiple symmetry-inequivalent images.
    structures: list[Structure] = []
    for doc in docs:
        if doc.structure is None:
            continue
        try:
            sga = SpacegroupAnalyzer(doc.structure, symprec=0.1)
            structures.append(sga.get_conventional_standard_structure())
        except Exception:
            structures.append(doc.structure)

    nn_finder = CrystalNN()
    seen: dict[str, Molecule] = {}

    for structure in structures:
        site_geometries = _classify_sites(structure)
        for mol in _polyhedra_from_structure(structure, nn_finder, site_geometries):
            if len(mol) > max_atoms:
                continue
            key = mol.composition.reduced_formula
            if key not in seen:
                seen[key] = mol

    return list(seen.values())
