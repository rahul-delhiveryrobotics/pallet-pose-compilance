"""Nominal pallet 3D model factory (design: Pallet physical dimensions, R13.3).

This module builds the known 3D pallet model used for known-geometry PnP. It
does **not** redefine the data model — it reuses the pydantic
:class:`~pallet_pose_compliance.output.schema.PalletModel` (the single source of
truth for the schema) and populates it with nominal dimensions.

Convention (design: Coordinate Systems and Conventions)
-------------------------------------------------------
The model is expressed in a **pallet-local frame** whose axes match Floor_Frame
conventions when the pallet is at ``theta = 0`` (long axis along ``+X``):

- Origin at the floor-projected pallet centre (the centroid of the four bottom
  deck corners), on the floor plane ``Z = 0``.
- ``+X`` along the pallet long axis (length), ``+Y`` along the width, ``+Z`` up.
- Bottom deck corners lie on ``Z = 0`` (floor contact). Top deck corners are
  elevated by ``deck_height`` so PnP uses their true height above the floor
  (R9.2, R9.4) rather than a floor homography on elevated points.

The default footprint is the standard warehouse pallet nominal spec
(1200 x 800 mm, ~144 mm deck height). Because these are nominal-spec values and
not a physically measured pallet, ``provenance`` is
:attr:`~pallet_pose_compliance.output.provenance.ProvenanceLabel.ESTIMATED`
(design: Pallet physical dimensions and source, R13.3, R26.1). The
``dim_uncertainty_m`` is carried into Monte Carlo propagation downstream.
"""

from __future__ import annotations

from ..output.provenance import ProvenanceLabel
from ..output.schema import DimensionsM, PalletModel

__all__ = [
    "NOMINAL_PALLET_LENGTH_M",
    "NOMINAL_PALLET_WIDTH_M",
    "NOMINAL_PALLET_DECK_HEIGHT_M",
    "NOMINAL_DIM_UNCERTAINTY_M",
    "NOMINAL_PALLET_SOURCE",
    "build_nominal_pallet_model",
]

# Standard warehouse pallet nominal footprint (EUR/ISO-style 1200 x 800 mm) and
# a representative full deck height. Values are nominal spec, hence `estimated`.
NOMINAL_PALLET_LENGTH_M: float = 1.200
NOMINAL_PALLET_WIDTH_M: float = 0.800
NOMINAL_PALLET_DECK_HEIGHT_M: float = 0.144

#: Nominal dimension uncertainty (metres, 1-sigma) fed into MC propagation.
#: Reflects tolerance/wear on a nominal-spec pallet whose exact build is unknown.
NOMINAL_DIM_UNCERTAINTY_M: float = 0.010

#: Documented source string for the nominal model (recorded in the PalletModel).
NOMINAL_PALLET_SOURCE: str = (
    "nominal standard warehouse pallet spec (1200x800 mm, ~144 mm deck height); "
    "not a physically measured pallet"
)


def _rectangle_corners(
    length_m: float, width_m: float, z_m: float
) -> list[list[float]]:
    """Return four ``[x, y, z]`` corners of a centred rectangle at height ``z``.

    Corners are ordered counter-clockwise viewed from ``+Z`` starting at the
    ``(+x, +y)`` corner, consistent with the right-handed Floor_Frame and the
    positive (CCW) orientation convention:

        0: (+L/2, +W/2)   1: (-L/2, +W/2)
        2: (-L/2, -W/2)   3: (+L/2, -W/2)
    """
    half_l = length_m / 2.0
    half_w = width_m / 2.0
    return [
        [+half_l, +half_w, z_m],
        [-half_l, +half_w, z_m],
        [-half_l, -half_w, z_m],
        [+half_l, -half_w, z_m],
    ]


def build_nominal_pallet_model(
    length_m: float = NOMINAL_PALLET_LENGTH_M,
    width_m: float = NOMINAL_PALLET_WIDTH_M,
    deck_height_m: float = NOMINAL_PALLET_DECK_HEIGHT_M,
    dim_uncertainty_m: float = NOMINAL_DIM_UNCERTAINTY_M,
    *,
    source: str = NOMINAL_PALLET_SOURCE,
    provenance: ProvenanceLabel = ProvenanceLabel.ESTIMATED,
) -> PalletModel:
    """Build a nominal :class:`PalletModel` from nominal-spec dimensions.

    The returned model is centred at the floor-projected pallet centre with the
    long axis along ``+X`` (``theta = 0``). Bottom corners lie on ``Z = 0`` and
    top corners are elevated by ``deck_height_m`` so PnP respects true keypoint
    height above the floor (R9.2, R9.4).

    Parameters
    ----------
    length_m, width_m, deck_height_m:
        Pallet nominal dimensions in metres. ``length`` is the long axis.
    dim_uncertainty_m:
        1-sigma dimension uncertainty fed into MC propagation (R13.3).
    source:
        Documented provenance/source string for the dimensions.
    provenance:
        Provenance of the dimensions. Defaults to ``ESTIMATED`` for a
        nominal-spec pallet; pass ``MEASURED`` only for a physically measured
        pallet (design: Pallet physical dimensions and source, R26.1).

    Returns
    -------
    PalletModel
        A schema-valid pallet model reusing the output-layer pydantic model.

    Raises
    ------
    ValueError
        If any dimension is non-positive or the uncertainty is negative.
    """
    if length_m <= 0.0 or width_m <= 0.0 or deck_height_m <= 0.0:
        raise ValueError(
            "pallet dimensions must be positive "
            f"(length={length_m}, width={width_m}, deck_height={deck_height_m})"
        )
    if dim_uncertainty_m < 0.0:
        raise ValueError(
            f"dim_uncertainty_m must be non-negative, got {dim_uncertainty_m}"
        )

    bottom_corners = _rectangle_corners(length_m, width_m, z_m=0.0)
    top_corners = _rectangle_corners(length_m, width_m, z_m=deck_height_m)

    return PalletModel(
        bottom_corners_m=bottom_corners,
        top_corners_m=top_corners,
        dimensions_m=DimensionsM(
            length=length_m,
            width=width_m,
            deck_height=deck_height_m,
        ),
        dim_uncertainty_m=dim_uncertainty_m,
        source=source,
        provenance=provenance,
    )
