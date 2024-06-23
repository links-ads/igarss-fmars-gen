import os
import shapely
import geopandas as gpd
from typing import Tuple, Optional

import functools
import glob
import gzip
import multiprocessing
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import geopandas as gpd
from IPython import display
import pandas as pd
#import s2geometry as s2
import s2cell as s2
import shapely
#import tensorflow as tf
import tqdm.notebook

data_type = 'polygons'
BUILDING_DOWNLOAD_PATH = (f'gs://open-buildings-data/v3/{data_type}_s2_level_6_gzip_no_header')

def get_filename_and_region_dataframe(your_own_wkt_polygon: str) -> Tuple[str, gpd.geodataframe.GeoDataFrame]:
    """Returns output filename and a geopandas dataframe with one region row."""
    data_type = 'polygons'
    filename = f'open_buildings_v3_{data_type}_your_own_wkt_polygon.csv.gz'
    region_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt([your_own_wkt_polygon]), crs='EPSG:4326')
    if not isinstance(region_df.iloc[0].geometry, shapely.geometry.polygon.Polygon) and \
        not isinstance(region_df.iloc[0].geometry, shapely.geometry.multipolygon.MultiPolygon):
        raise ValueError("`your_own_wkt_polygon` must be a POLYGON or MULTIPOLYGON.")
    print(f'Preparing your_own_wkt_polygon.')
    return filename, region_df

def get_bounding_box_s2_covering_tokens(region_geometry: shapely.geometry.base.BaseGeometry) -> List[str]:
    region_bounds = region_geometry.bounds
    s2_lat_lng_rect = s2.S2LatLngRect_FromPointPair(s2.S2LatLng_FromDegrees(region_bounds[1], region_bounds[0]), 
                                                    s2.S2LatLng_FromDegrees(region_bounds[3], region_bounds[2]))
    coverer = s2.S2RegionCoverer()
    # NOTE: Should be kept in-sync with s2 level in BUILDING_DOWNLOAD_PATH.
    coverer.set_fixed_level(6)
    coverer.set_max_cells(1000000)
    return [cell.ToToken() for cell in coverer.GetCovering(s2_lat_lng_rect)]

def s2_token_to_shapely_polygon(s2_token: str) -> shapely.geometry.polygon.Polygon:
    s2_cell = s2.S2Cell(s2.S2CellId_FromToken(s2_token, len(s2_token)))
    coords = []
    for i in range(4):
        s2_lat_lng = s2.S2LatLng(s2_cell.GetVertex(i))
        coords.append((s2_lat_lng.lng().degrees(), s2_lat_lng.lat().degrees()))
    return shapely.geometry.Polygon(coords)

def download_s2_token(s2_token: str, region_df: gpd.geodataframe.GeoDataFrame) -> Optional[str]:
    """Downloads the matching CSV file with polygons for the `s2_token`.

    NOTE: Only polygons inside the region are kept.
    NOTE: Passing output via a temporary file to reduce memory usage.

    Args:
    s2_token: S2 token for which to download the CSV file with building polygons.
            The S2 token should be at the same level as the files in BUILDING_DOWNLOAD_PATH.
    region_df: A geopandas dataframe with only one row that contains the region for which to keep polygons.

    Returns:
    Either filepath which contains a gzipped CSV without header for the `s2_token` subfiltered to only contain building polygons inside the region
    or None which means that there were no polygons inside the region for this `s2_token`.
    """
    s2_cell_geometry = s2_token_to_shapely_polygon(s2_token)
    region_geometry = region_df.iloc[0].geometry
    prepared_region_geometry = shapely.prepared.prep(region_geometry)
    # If the s2 cell doesn't intersect the country geometry at all then we can
    # know that all rows would be dropped so instead we can just return early.
    if not prepared_region_geometry.intersects(s2_cell_geometry):
        return None
    try:
        # Using tf.io.gfile.GFile gives better performance than passing the GCS path
        # directly to pd.read_csv.
        #with tf.io.gfile.GFile(os.path.join(BUILDING_DOWNLOAD_PATH, f'{s2_token}_buildings.csv.gz'), 'rb') as gf:
        with open(os.path.join(BUILDING_DOWNLOAD_PATH, f'{s2_token}_buildings.csv.gz'), 'rb') as gf:
            # If the s2 cell is fully covered by country geometry then can skip
            # filtering as we need all rows.
            if prepared_region_geometry.covers(s2_cell_geometry):
                with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp_f:
                    shutil.copyfileobj(gf, tmp_f)
                    return tmp_f.name
            # Else take the slow path.
            # NOTE: We read in chunks to save memory.
            csv_chunks = pd.read_csv(gf, chunksize=2000000, dtype=object, compression='gzip', header=None)
            tmp_f = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
            tmp_f.close()
            for csv_chunk in csv_chunks:
                points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(csv_chunk[1], csv_chunk[0]), crs='EPSG:4326')
                # sjoin 'within' was faster than using shapely's 'within' directly.
                points = gpd.sjoin(points, region_df, predicate='within')
                csv_chunk = csv_chunk.iloc[points.index]
                csv_chunk.to_csv(
                    tmp_f.name,
                    mode='ab',
                    index=False,
                    header=False,
                    compression={
                        'method': 'gzip',
                        'compresslevel': 1
                    })
            return tmp_f.name
        
    except FileNotFoundError:
        return None