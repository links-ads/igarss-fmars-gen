import os
import json
import shapely
from typing import List
from pathlib import Path
import numpy as np
import geopandas as gpd
from maxarseg.geo_datasets import geoDatasets
from shapely.geometry.polygon import Polygon
from shapely import geometry
import rasterio
import pandas as pd
import glob
import scipy

#from maxarseg import segment

def path_2_tile_aoi(tile_path, root = './metadata/from_github_maxar_metadata/datasets' ):
    """
    Create a shapely Polygon from a tile_path
    Example of a tile_path: '../Gambia-flooding-8-11-2022/pre/10300100CFC9A500/033133031213.tif'
    """
    if isinstance(tile_path, str):
        event = tile_path.split('/')[-4]
        child = tile_path.split('/')[-2]
        tile = tile_path.split('/')[-1].replace(".tif", "")
    elif isinstance(tile_path, Path):
        event = tile_path.parts[-4]
        child = tile_path.parts[-2]
        tile = tile_path.parts[-1].replace(".tif", "")
    else:
        raise TypeError("tile_path must be a string or a Path object")
    
    try:
        path_2_child_geojson = os.path.join(root, event, child +'.geojson')
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    except:
        file_pattern = str(os.path.join(root, event, child + '*inv.geojson'))
        file_list = glob.glob(f"{file_pattern}")
        assert len(file_list) == 1, f"Found {len(file_list)} files with pattern {file_pattern}. Expected 1 file."
        path_2_child_geojson = file_list[0]
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    
    
    j = [el["properties"]["proj:geometry"] for el in child_geojson['features'] if el['properties']['quadkey'] == tile][0]
    tile_polyg = shapely.geometry.shape(j)
    return tile_polyg

def path_2_tile_aoi_no_water(tile_path, land_gdf = None, root = './metadata/from_github_maxar_metadata/datasets' ):
    """
    Create a shapely Polygon from a tile_path
    Example of a tile_path: '../Gambia-flooding-8-11-2022/pre/10300100CFC9A500/033133031213.tif'
    """
    if isinstance(tile_path, str):
        event = tile_path.split('/')[-4]
        child = tile_path.split('/')[-2]
        tile = tile_path.split('/')[-1].replace(".tif", "")
    elif isinstance(tile_path, Path):
        event = tile_path.parts[-4]
        child = tile_path.parts[-2]
        tile = tile_path.parts[-1].replace(".tif", "")
    else:
        raise TypeError("tile_path must be a string or a Path object")
    
    try:
        path_2_child_geojson = os.path.join(root, event, child +'.geojson')
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    except:
        file_pattern = str(os.path.join(root, event, child + '*inv.geojson'))
        file_list = glob.glob(f"{file_pattern}")
        assert len(file_list) == 1, f"Found {len(file_list)} files with pattern {file_pattern}. Expected 1 file."
        path_2_child_geojson = file_list[0]
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    
    prj_crs = [el['properties']['proj:epsg'] for el in child_geojson['features'] if el['properties']['quadkey'] == tile][0]
    j = [el["geometry"] for el in child_geojson['features'] if el['properties']['quadkey'] == tile][0]
    tile_polyg = shapely.geometry.shape(j)
    
    tile_adj_aois = []
    if land_gdf is None: #caso in cui tutto l'evento non interseca confine wl
        tile_adj_aois.append(tile_polyg)
    else:
        intersection_gdf = land_gdf.intersection(tile_polyg).loc[lambda x: ~x.is_empty]
        if len(intersection_gdf) == 0:
            print('Tile non interseca land. Solo mare. Mask vuota')
            tile_adj_aois.append(Polygon())
        else:
            if land_gdf.contains(tile_polyg).any():
                print('Completely contained in land. No mod to tile_aoi')
                tile_adj_aois.append(tile_polyg)
            else:
                print('Tile interseca wlb')
                for geom in intersection_gdf:
                    tile_adj_aois.append(geom)
                    
    return gpd.GeoDataFrame(geometry = tile_adj_aois, crs="EPSG:4326").to_crs(prj_crs)

def boundingBox_2_Polygon(bounding_box):
    """
    Create a shapely Polygon from a BoundingBox
    """
    minx, miny, maxx, maxy = bounding_box.minx, bounding_box.miny, bounding_box.maxx, bounding_box.maxy
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    bbox_polyg = shapely.geometry.Polygon(vertices)
    return bbox_polyg

def xyxy_2_Polygon(xyxy_box):
    """
    Create a shapely Polygon from a xyxy box
    """
    if not len(xyxy_box) == 4: #allow for a tuple of 2 points. E.g. ((minx, miny), (maxx, maxy))
        minx, miny = xyxy_box[0]
        maxx, maxy = xyxy_box[1]
    else:    
        minx, miny, maxx, maxy = xyxy_box
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return shapely.geometry.Polygon(vertices)

def xyxyBox2Polygon(xyxy_box):
    """
    Create a shapely Polygon from a xyxy box
    """
    minx, miny, maxx, maxy = xyxy_box
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    bbox_polyg = shapely.geometry.Polygon(vertices)
    return bbox_polyg

def boundingBox_2_centralPoint(bounding_box):
    """
    Create a shapely Point from a BoundingBox
    """
    minx, miny, maxx, maxy = bounding_box.minx, bounding_box.miny, bounding_box.maxx, bounding_box.maxy
    return shapely.geometry.Point((minx + maxx)/2, (miny + maxy)/2)

def align_bbox(bbox: Polygon):
    """
    Turn the polygon into a bbox axis aligned
    """
    minx, miny, maxx, maxy = bbox.bounds
    return minx, miny, maxx, maxy

def rel_bbox_coords(geodf:gpd.GeoDataFrame,
                    ref_coords:tuple,
                    res,
                    ext_mt = None):
    """
    Returns the relative coordinates of a bbox w.r.t. a reference bbox in the 'geometry' column.
    Goes from absolute geo coords to relative coords in the image.

    Inputs:
        geodf: dataframe with bboxes
        ref_coords: a tuple in the format (minx, miny, maxx, maxy)
        res: resolution of the image
        ext_mt: meters to add to each edge of the box (the center remains fixed)
    Returns:
        a list of tuples with the relative coordinates of the bboxes [(minx, miny, maxx, maxy), ...]
    """
    result = []
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner of the patch sample extracted from the tile
    #print('\nref_coords top left: ', ref_minx, ref_maxy )
    for geom in geodf.geometry:
        minx, miny, maxx, maxy = align_bbox(geom)
        if ext_mt != None or ext_mt != 0:
            minx -= (ext_mt / 2)
            miny -= (ext_mt / 2)
            maxx += (ext_mt / 2)
            maxy += (ext_mt / 2)

        rel_bbox_coords = list(np.array([minx - ref_minx, ref_maxy - maxy, maxx - ref_minx, ref_maxy - miny]) / res)
        result.append(rel_bbox_coords)
    
    return result

def tile_sizes(dataset: geoDatasets.MxrSingleTile):
    """
    Returns the sizes of the tile given the path
    It uses the 
    """
    bounds = dataset.bounds
    x_size_pxl = (bounds.maxy - bounds.miny) / dataset.res
    y_size_pxl = (bounds.maxx - bounds.minx) / dataset.res
    
    if x_size_pxl % 1 != 0 or y_size_pxl % 1 != 0:
        raise ValueError("The sizes of the tile are not integers")
    
    return (int(x_size_pxl), int(y_size_pxl))

def tile_path_2_tile_size(tile_path):
    """
    Returns the sizes of the tile given the path
    """
    with rasterio.open(tile_path) as src:
        return src.width, src.height

def double_tuple_box_2_shapely_box(double_tuple_box):
    """
    Create a shapely Polygon from a double tuple box
    """
    minx, miny = double_tuple_box[0]
    maxx, maxy = double_tuple_box[1]
    return geometry.box(minx, miny, maxx, maxy)

def road_gdf_vs_aois_gdf(proj_road_gdf, aois_gdf):
    #Could be usefull but not used
    num_roads = len(proj_road_gdf)
    num_hits = np.array([0]*num_roads)
    in_aoi_roads_gdf = gpd.GeoSeries()
    for geom in aois_gdf.geometry:
        intersec_geom = proj_road_gdf.intersection(geom)
        valid_gdf = intersec_geom[~intersec_geom.is_empty]
        num_hits = num_hits + (~intersec_geom.is_empty.values)
        in_aoi_roads_gdf = gpd.GeoSeries(pd.concat([valid_gdf, in_aoi_roads_gdf], ignore_index=True))
        
    if any(num_hits > 1):
        raise NotImplementedError("Error: case in which a road is located in more than one area of interest. Not implemented.")
    else:
        return in_aoi_roads_gdf
    
def filter_road_gdf_vs_aois_gdf(proj_road_gdf, aois_gdf):
    num_roads = len(proj_road_gdf)
    num_hits = np.array([0]*num_roads)
    for geom in aois_gdf.geometry:
        hits = proj_road_gdf.intersects(geom)
        num_hits = num_hits + hits.values
    return proj_road_gdf[num_hits >= 1]

def intersection_road_gdf_vs_aois_gdf(proj_road_gdf, aois_gdf):
    intersected_roads = gpd.GeoSeries()
    num_roads = len(proj_road_gdf)
    for geom in aois_gdf.geometry:
        intersec_geom = proj_road_gdf.intersection(geom)
        valid_gdf = intersec_geom[~intersec_geom.is_empty]
        intersected_roads = gpd.GeoSeries(pd.concat([valid_gdf, intersected_roads], ignore_index=True))
        
    return intersected_roads

def entropy_from_lbl(lbl):
    flat_array = lbl.flatten()
    class_imp = []
    for i in [0, 1, 2, 255]:
        class_imp.append(np.sum(flat_array == i))
    return scipy.stats.entropy(class_imp, base = 2)

def compute_entropy_matrix(img, size = 1024):
    if len(img.shape) == 3:
        img = img.squeeze()
    entropy_matrix = np.zeros((int(img.shape[0]/size), int(img.shape[1]/size)))
    for i in range(int(img.shape[0]/size)):
        for j in range(int(img.shape[1]/size)):
            patch = img[i*size:(i+1)*size, j*size:(j+1)*size]
            entropy_matrix[i, j] = entropy_from_lbl(patch)
    return entropy_matrix