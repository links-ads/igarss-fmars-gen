from pathlib import Path
import sys
import geopandas as gpd
from maxarseg.build import geoDatasets
from pyquadkey2 import quadkey
import pandas as pd
from shapely import geometry
import json
from typing import List, Tuple, Union
import time
import os
from torchgeo.datasets import stack_samples
from maxarseg import samplers
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from maxarseg.segment import segment
import torch
import rasterio
from rasterio.features import rasterize

###########################
# Retrieve bbox coordinates 
###########################

def old_get_bbox_roads(mosaic_bbox: Union[List[Tuple], Tuple[Tuple]], region_name, roads_root = '/nfs/projects/overwatch/maxar-segmentation/microsoft-roads'):
    """
    Get a gdf containing the roads that intersect the mosaic_bbox.
    Input:
        mosaic_bbox: Bounding box of the mosaic in format (lon, lat). Example: ((-16.5, 13.5), (-15.5, 14.5))
        region_name: Name of the region. Example: 'AfricaWest-Full'
        roads_root: Root directory of the roads datasets
    """
    if region_name[-4:] != '.tsv':
        region_name = region_name + '.tsv'

    roads_root = Path(roads_root)
    road_df = pd.read_csv(roads_root/region_name, names =['country', 'geometry'], sep='\t')
    road_df['geometry'] = road_df['geometry'].apply(json.loads).apply(lambda d: geometry.shape(d.get('geometry')))
    road_gdf = gpd.GeoDataFrame(road_df, crs=4326)
    
    (minx, miny), (maxx, maxy) = mosaic_bbox
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)] #lon lat
    query_bbox_poly = geometry.Polygon(vertices)
    
    hits = road_gdf.geometry.intersects(query_bbox_poly)
    
    return road_gdf[hits]



