from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)


import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import box
import matplotlib.patches as patches


# ---- Types & simple data structures -------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in WGS84 (EPSG:4326).

    Attributes:
        minx: Minimum longitude.
        miny: Minimum latitude.
        maxx: Maximum longitude.
        maxy: Maximum latitude.
    """

    minx: float
    miny: float
    maxx: float
    maxy: float

    def to_geodataframe(self) -> GeoDataFrame:
        """Return the bounding box as a single-row GeoDataFrame with EPSG:4326."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [box(self.minx, self.miny, self.maxx, self.maxy)]},
            crs="EPSG:4326",
        )
        return gdf


LABELS: Tuple[str, ...] = ("bg", "road", "rail", "landscape", "land")


# ---- I/O helpers --------------------------------------------------------------------


def _city_data_folder(foldernames: Mapping[str, str], city: str) -> Path:
    """Build the path to the city's data folder."""
    try:
        return Path("data") / foldernames[city]
    except KeyError as exc:
        raise KeyError(f"City '{city}' not found in foldernames mapping.") from exc


def _read_shapefile_masked(filepath: Path, mask: GeoDataFrame) -> GeoDataFrame:
    """Read a shapefile and spatially mask it by the provided mask (bounding box)."""
    return gpd.read_file(str(filepath), mask=mask)


def local_albers_equal_area(minx, miny, maxx, maxy):
    """
    Construct a city-centered Albers Equal Area (AEA) CRS for accurate areas.
    Uses parallels at lat0 ± 10° which works well for small-to-medium regions.
    Returns a PROJ string accepted by GeoPandas/pyproj.
    """
    lat0 = 0.5 * (miny + maxy)  # latitude of origin (city mid-lat)
    lon0 = 0.5 * (minx + maxx)  # central meridian (city mid-lon)
    lat1 = lat0 - 10.0  # first standard parallel
    lat2 = lat0 + 10.0  # second standard parallel
    return (
        f"+proj=aea +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} "
        f"+datum=WGS84 +units=m +no_defs +type=crs"
    )


def plot_bounding_boxes(
    bounding_boxes,
):

    # --- Build polygons in WGS84 --------------------------------------------------------
    records = []
    for city, b in bounding_boxes.items():
        geom = box(b["minx"], b["miny"], b["maxx"], b["maxy"])
        records.append({"city": city, "geometry": geom, **b})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # --- Reproject each bbox to its own local AEA, measure, and plot --------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    x_offset_m = 0.0
    gap_m = 10_000  # 10 km between rectangles

    for _, row in gdf.iterrows():
        city = row["city"]
        # Build a 1-row GeoDataFrame for this city and project to its local equal-area CRS
        proj = local_albers_equal_area(
            row["minx"], row["miny"], row["maxx"], row["maxy"]
        )
        g1 = gpd.GeoDataFrame(
            [{"city": city, "geometry": row["geometry"]}], crs=gdf.crs
        ).to_crs(proj)

        # Get width/height from projected bounds (meters)
        minx_p, miny_p, maxx_p, maxy_p = g1.total_bounds
        width_m = maxx_p - minx_p
        height_m = maxy_p - miny_p

        # Draw a rectangle with the exact projected width/height (area true)
        rect = patches.Rectangle(
            (x_offset_m, 0.0),
            width_m,
            height_m,
            edgecolor="blue",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Label above
        ax.text(
            x_offset_m + width_m / 2.0,
            height_m + (0.02 * max(height_m, 1.0)),
            city,
            ha="center",
            fontsize=10,
        )

        x_offset_m += width_m + gap_m

    # Make area/shape visually honest
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-0.05 * x_offset_m, x_offset_m + 0.05 * x_offset_m)
    ax.set_ylim(
        0, ax.get_ylim()[1] * 1.1 if ax.get_ylim()[1] > 0 else max(1000, height_m * 1.1)
    )
    ax.axis("off")
    ax.set_title("Bounding box sizes (equal-area per-city projection, meters)")


# ---- Loading ------------------------------------------------------------------------


def load_city_layers(
    foldernames: Mapping[str, str],
    bounding_boxes: Mapping[str, Mapping[str, float]],
    city: str,
) -> Tuple[
    Tuple[
        GeoDataFrame,  # railroads
        GeoDataFrame,  # normal_roads
        GeoDataFrame,  # primary_roads
        GeoDataFrame,  # water (areas)
        GeoDataFrame,  # landuse_filtered
        GeoDataFrame,  # waterways (lines)
    ],
    Optional[GeoDataFrame],  # clipped geoboundaries (Istanbul only)
    BoundingBox,  # bbox values
]:
    """Load and pre-filter geospatial layers for a given city.

    Args:
        foldernames: Mapping of city name to relative data folder name under `data/`.
        bounding_boxes: Mapping of city name to a dict with keys 'minx', 'miny', 'maxx', 'maxy'.
        city: City key present in both mappings.

    Returns:
        A tuple of:
            - railroads
            - normal_roads
            - primary_roads (subset of normal_roads where fclass == 'primary')
            - water (areas)
            - landuse_filtered (subset by fclass in {'forest','park','farmland','meadow','allotments'})
            - waterways (lines)
            - clipped_geoboundaries (if city == 'Istanbul', else None)
            - bounding box values as `BoundingBox`.

    Raises:
        KeyError: If the city is missing from the provided mappings.
        FileNotFoundError: If any expected file is missing.
    """
    shp_folder = _city_data_folder(foldernames, city)
    bbox_vals = bounding_boxes.get(city)
    if bbox_vals is None:
        raise KeyError(f"City '{city}' not found in bounding_boxes mapping.")
    bbox = BoundingBox(**bbox_vals)
    bbox_gdf = bbox.to_geodataframe()

    # Expected shapefiles
    railroads_path = shp_folder / "gis_osm_railways_free_1.shp"
    roads_path = shp_folder / "gis_osm_roads_free_1.shp"
    landuse_path = shp_folder / "gis_osm_landuse_a_free_1.shp"
    waterways_path = shp_folder / "gis_osm_waterways_free_1.shp"
    water_path = shp_folder / "gis_osm_water_a_free_1.shp"

    for p in (railroads_path, roads_path, landuse_path, waterways_path, water_path):
        if not p.exists():
            raise FileNotFoundError(f"Required shapefile not found: {p}")

    # Load masked layers
    railroads = _read_shapefile_masked(railroads_path, bbox_gdf)
    normal_roads = _read_shapefile_masked(roads_path, bbox_gdf)
    primary_roads = normal_roads.loc[normal_roads.get("fclass") == "primary"].copy()
    water = _read_shapefile_masked(water_path, bbox_gdf)
    landuse = _read_shapefile_masked(landuse_path, bbox_gdf)
    waterways = _read_shapefile_masked(waterways_path, bbox_gdf)

    # Filter landuse by fclass
    landuse_filtered = landuse[
        landuse.get("fclass").isin(
            ["forest", "park", "farmland", "meadow", "allotments"]
        )
    ].copy()

    # Optional: country geoboundaries for Istanbul only
    clipped_geoboundaries: Optional[GeoDataFrame]
    if city == "Istanbul":
        geoboundaries_path = (
            Path("data")
            / Path("geoBoundaries-TUR-ADM0-all")
            / "geoBoundaries-TUR-ADM0.shp"
        )
        if not geoboundaries_path.exists():
            raise FileNotFoundError(
                f"Geoboundaries shapefile not found: {geoboundaries_path}"
            )

        geoboundaries = gpd.read_file(str(geoboundaries_path))
        if geoboundaries.crs != "EPSG:4326":
            geoboundaries = geoboundaries.to_crs("EPSG:4326")
        clipped_geoboundaries = gpd.overlay(geoboundaries, bbox_gdf, how="intersection")
    else:
        clipped_geoboundaries = None

    return (
        (railroads, normal_roads, primary_roads, water, landuse_filtered, waterways),
        clipped_geoboundaries,
        bbox_vals,
    )


# ---- Plotting & saving --------------------------------------------------------------


def plot_bbox_outline(bbox: dict, ax, color="red", square=False, linewidth=2):
    """Draw bounding box or square around given bbox."""
    minx, miny, maxx, maxy = bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
    width, height = maxx - minx, maxy - miny

    if square:
        # Center the square on the bbox
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        side = max(width, height)
        minx, maxx = cx - side / 2, cx + side / 2
        miny, maxy = cy - side / 2, cy + side / 2

    geom = box(minx, miny, maxx, maxy)
    gpd.GeoSeries([geom]).plot(ax=ax, facecolor=color, linewidth=linewidth)


def plot_city_map(
    shapefiles: Tuple[GeoDataFrame],
    bbox: dict,
    colors: Mapping[str, str],
    fig,
    ax,
    clipped_geoboundaries: Optional[GeoDataFrame] = None,
    rotate: bool = False,
    # figsize: Tuple[float, float] = (12, 12),
) -> plt.Axes:
    """Plot the city map for a given color assignment.

    Args:
        railroads: Railways layer.
        primary_roads: Primary roads subset.
        water: Water areas layer.
        landuse_filtered: Landuse polygons filtered by allowed classes.
        waterways: Waterways lines.
        bbox: BoundingBox to set plot limits.
        colors: Mapping of label -> hex color. Expected labels: ('bg','road','rail','landscape','land').
        clipped_geoboundaries: Optional country boundaries clipped to the bbox (Istanbul).
        figsize: Matplotlib figure size (in inches).

    Returns:
        The matplotlib Axes used for the plot.

    Raises:
        ValueError: If required color labels are missing.
    """

    (railroads, normal_roads, primary_roads, water, landuse_filtered, waterways) = (
        shapefiles
    )

    missing = [label for label in LABELS if label not in colors]
    if missing:
        raise ValueError(f"Missing colors for labels: {missing}. Expected {LABELS}")

    # fig, ax = plt.subplots(figsize=figsize)
    bg_color = colors["bg"]
    road_color = colors["road"]
    rail_color = colors["rail"]
    landscape_color = colors["landscape"]
    land_color = colors["land"]
    water_color = colors["bg"]

    # If we have Istanbul geoboundaries, draw land; otherwise use land as background.
    if clipped_geoboundaries is not None:
        plot_bbox_outline(bbox, ax, color=water_color, square=False)

        clipped_geoboundaries.plot(ax=ax, facecolor=land_color, linewidth=0)

    else:
        # Use land color as overall background if no geoboundary
        # fig.patch.set_facecolor(land_color)
        # ax.set_facecolor(land_color)
        bg_color = land_color  # keep consistency for saving facecolor

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Layers (order matters)
    landuse_filtered.plot(ax=ax, color=landscape_color)
    waterways.plot(ax=ax, color=water_color, linewidth=0.5)
    water.plot(ax=ax, color=water_color)
    primary_roads.plot(ax=ax, color=road_color, linewidth=0.2)
    # normal_roads.plot(ax=ax, color=road_color, linewidth=0.1)
    railroads.plot(ax=ax, color=rail_color, linewidth=0.3)

    # if rotate:
    #     # Swap axes to rotate the view 90° counterclockwise
    #     ax.set_xlim(bbox["miny"], bbox["maxy"])
    #     ax.set_ylim(bbox["maxx"], bbox["minx"])
    #     ax.invert_yaxis()  # keep orientation consistent
    # else:
    ax.set_xlim(bbox["minx"], bbox["maxx"])
    ax.set_ylim(bbox["miny"], bbox["maxy"])

    ax.axis("off")

    # Store the effective facecolor used for export on the Axes for convenience
    ax._export_facecolor = bg_color  # type: ignore[attr-defined]
    return ax


def save_axes_as_pdf(ax: plt.Axes, out_path: Path) -> None:
    """Save a matplotlib Axes' figure to a PDF file.

    Args:
        ax: Matplotlib Axes previously created by `plot_city_map`.
        out_path: Output path ending with `.pdf`.
    """
    folder = os.path.dirname(str(out_path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    facecolor = getattr(ax, "_export_facecolor", None) or ax.figure.get_facecolor()
    ax.figure.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=facecolor,
    )
    plt.close(ax.figure)


# ---- Public API ---------------------------------------------------------------------


def save_file(
    foldername_out: str,
    foldernames: Mapping[str, str],
    bounding_boxes: Mapping[str, Mapping[str, float]],
    city: str,
    color_sets: Sequence[Sequence[str]],
) -> None:
    """Render and save PDF maps for a city using multiple color sets.

    This function loads the required geospatial layers, iterates over the given
    color sets, assigns colors to the expected labels, and writes one PDF per
    color set to `foldername_out`.

    Args:
        foldername_out: Output folder or filename prefix. The function will write
            files named `<foldername_out><city>_<idx>.pdf`. If `foldername_out` is a
            folder, include a trailing slash.
        foldernames: Mapping of city name to relative data folder name under `data/`.
        bounding_boxes: Mapping of city name to a dict with keys 'minx', 'miny', 'maxx', 'maxy'.
        city: City key present in both mappings.
        color_sets: Iterable of color sequences (length 5), matching LABELS order:
            ('bg','road','rail','landscape','land').

    Raises:
        ValueError: If any color set does not have exactly 5 colors.
        KeyError, FileNotFoundError: From underlying loaders if inputs are missing.
    """
    (
        shapefiles,
        clipped_geoboundaries,
        bbox,
    ) = load_city_layers(foldernames, bounding_boxes, city)
    fig, ax = plt.subplots(figsize=(12, 12))
    out_prefix = Path(foldername_out)
    # Validate and iterate color sets
    for idx, color_set in enumerate(color_sets):
        if len(color_set) != len(LABELS):
            raise ValueError(
                f"Color set at index {idx} has length {len(color_set)}; expected {len(LABELS)}."
            )
        colors = {label: color_set[i] for i, label in enumerate(LABELS)}

        ax = plot_city_map(
            shapefiles=shapefiles,
            bbox=bbox,
            colors=colors,
            fig=fig,
            ax=ax,
            clipped_geoboundaries=clipped_geoboundaries,
        )
        out_path = out_prefix.parent / f"{out_prefix.name}{city}_{idx}.pdf"
        save_axes_as_pdf(ax, out_path)


def iterate_colorsets(
    colorsets: Sequence[Sequence[str]], n_iterations: int
) -> List[List[str]]:
    """Return shuffled variants of each provided color set.

    For each `color_set` in `colorsets`, produce `n_iterations` shuffled copies.

    Args:
        colorsets: Sequence of color sets; each should be a sequence of color strings.
        n_iterations: Number of shuffled variants to produce per color set.

    Returns:
        A flat list of shuffled color sets (as lists).

    Raises:
        ValueError: If any color set is empty or if `n_iterations` < 1.
    """
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1.")
    random_colorsets: List[List[str]] = []
    for color_set in colorsets:
        if not color_set:
            raise ValueError("Encountered an empty color set.")
        for _ in range(n_iterations):
            shuffled = list(color_set)
            random.shuffle(shuffled)
            random_colorsets.append(shuffled)
    return random_colorsets


### NEW CODE ############


# A0 size (inches)
A0_WIDTH_IN = 33.11
A0_HEIGHT_IN = 46.81

# Approx meters per degree latitude
M_PER_DEG_LAT = 111_320.0


def _bbox_size_meters(bbox: dict) -> tuple[float, float]:
    """Approx width/height in meters from EPSG:4326 bbox (cos(lat) correction)."""
    dlon = bbox["maxx"] - bbox["minx"]
    dlat = bbox["maxy"] - bbox["miny"]
    lat_mid = 0.5 * (bbox["miny"] + bbox["maxy"])
    width_m = dlon * M_PER_DEG_LAT * math.cos(math.radians(lat_mid))
    height_m = dlat * M_PER_DEG_LAT
    return max(width_m, 1e-6), max(height_m, 1e-6)


def save_a0_three_axes_same_scale(
    *,
    cities: list[str],  # e.g. ["Tehran", "Darmstadt", "Aachen"]
    foldernames: Mapping[str, str],
    bounding_boxes: Mapping[str, Mapping[str, float]],
    colors_per_city: Mapping[str, Mapping[str, str]],
    out_pdf: str,
    dpi: int = 300,
    vertical_gap_m: float = 1_000.0,  # 10 km between axes
    margins_in: float = 0.75,
    portrait: bool = True,
):
    """Create a DIN A0 PDF with three axes, each sized to its city's real dimensions."""
    # assert len(cities) == 3, "This helper expects exactly three cities."

    n_cities = len(cities)

    # Load once
    items = []
    for city in cities:
        shapefiles, clipped, bbox = load_city_layers(foldernames, bounding_boxes, city)
        w_m, h_m = _bbox_size_meters(bbox)
        items.append(
            {
                "city": city,
                "shapefiles": shapefiles,
                "clipped": clipped,
                "bbox": bbox,
                "w_m": w_m,
                "h_m": h_m,
                "colors": colors_per_city[city],
            }
        )

    # Layout in portrait: stack vertically, same page scale
    W_in, H_in = (
        (A0_WIDTH_IN, A0_HEIGHT_IN) if portrait else (A0_HEIGHT_IN, A0_WIDTH_IN)
    )
    avail_w_in = W_in - 2 * margins_in
    avail_h_in = H_in - 2 * margins_in

    max_width_m = max(it["w_m"] for it in items)
    total_height_m = sum(it["h_m"] for it in items) + vertical_gap_m * (
        n_cities - 1
    )  # n items -> n-1  gaps

    # One shared scale (inches per meter)
    scale_in_per_m = min(avail_w_in / max_width_m, avail_h_in / total_height_m)

    # Convert meter sizes to inches for axes rectangles
    gap_in = vertical_gap_m * scale_in_per_m
    for it in items:
        it["w_in"] = it["w_m"] * scale_in_per_m
        it["h_in"] = it["h_m"] * scale_in_per_m

    # Build figure
    fig = plt.figure(figsize=(W_in, H_in), dpi=dpi)
    current_y_in = margins_in
    current_x_in = margins_in
    current_x_in2 = margins_in

    # Create each axes with appropriate physical size, centered horizontally
    for idx, it in enumerate(items):
        if idx == 1:
            rotate = True
        else:
            rotate = False

        if idx <= 1:
            left_in = current_x_in
            bottom_in = margins_in
            ax = fig.add_axes(
                [
                    left_in / W_in,
                    bottom_in / H_in,
                    it["w_in"] / W_in,
                    it["h_in"] / H_in,
                ]
            )

            # Draw this city onto its own axes
            plot_city_map(
                shapefiles=it["shapefiles"],
                bbox=it["bbox"],
                colors=it["colors"],
                clipped_geoboundaries=it["clipped"],
                fig=fig,
                ax=ax,
                rotate=rotate,
            )

            current_y_in = max(
                current_y_in,
                margins_in + it["h_in"] + (gap_in if idx < len(items) - 1 else 0.0),
            )
            current_x_in += it["w_in"] + (gap_in if idx < len(items) - 1 else 0.0)
        else:
            left_in = current_x_in2
            bottom_in = current_y_in
            ax = fig.add_axes(
                [
                    left_in / W_in,
                    bottom_in / H_in,
                    it["w_in"] / W_in,
                    it["h_in"] / H_in,
                ]
            )
            # Draw this city onto its own axes
            plot_city_map(
                shapefiles=it["shapefiles"],
                bbox=it["bbox"],
                colors=it["colors"],
                clipped_geoboundaries=it["clipped"],
                fig=fig,
                ax=ax,  # <-- use our axes
            )

            current_x_in2 += it["w_in"] + (gap_in if idx < len(items) - 1 else 0.0)

    # Choose a uniform page background (e.g., land color of the first city)
    page_face = items[0]["colors"]["land"]
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=page_face)
    plt.close(fig)
