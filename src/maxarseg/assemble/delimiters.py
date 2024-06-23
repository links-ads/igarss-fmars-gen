from pathlib import Path
import sys
import geopandas as gpd
from typing import List, Tuple, Union
from maxarseg.assemble import names
import pyproj
import glob

def get_mosaic_bbox(event_name, mosaic_name, path_mosaic_metatada = './metadata/from_github_maxar_metadata/datasets', extra_mt = 0, return_proj_coords = False):
    """
    Get the bbox of a mosaic. It return the coordinates of the bottom left and top right corners.
    Input:
        event_name: Example: 'Gambia-flooding-8-11-2022'
        mosaic_name: It could be an element of the output of get_mosaics_names(). Example: '104001007A565700'
        path_mosaic_metatada: Path to the folder containing the geojson
        extra_mt: Extra meters added to all bbox sides. The center of the bbox remanis the same. (To be sure all elements are included)
        return_proj_coords: If True, it returns the coordinates in the projection of the mosaic.
    Output:
        pair of cordinates in format (lon, lat) or (x, y) if return_proj_coords is True
    """
    path_mosaic_metatada = Path(path_mosaic_metatada)
    file_name = mosaic_name + '.geojson'
    geojson_path = path_mosaic_metatada / event_name / file_name
    try:
        gdf = gpd.read_file(geojson_path)
    except:
        file_pattern = str(path_mosaic_metatada / event_name /mosaic_name) + '*inv.geojson'
        file_list = glob.glob(f"{file_pattern}")
        assert len(file_list) == 1, f"Found {len(file_list)} files with pattern {file_pattern}. Expected 1 file."
        gdf = gpd.read_file(file_list[0])
        
    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0

    for _, row in gdf.iterrows():
        tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = [float(el) for el in row['proj:bbox'].split(',')]
        if tmp_minx < minx:
            minx = tmp_minx
        if tmp_miny < miny:
            miny = tmp_miny
        if tmp_maxx > maxx:
            maxx = tmp_maxx
        if tmp_maxy > maxy:
            maxy = tmp_maxy

    #enlarge bbox
    minx -= (extra_mt/2)
    miny -= (extra_mt/2)
    maxx += (extra_mt/2)
    maxy += (extra_mt/2)
    if not return_proj_coords:
        source_crs = gdf['proj:epsg'].values[0]
        target_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        bott_left_lat, bott_left_lon = transformer.transform(minx, miny)
        top_right_lat, top_right_lon = transformer.transform(maxx, maxy)
        
        return ((bott_left_lon, bott_left_lat), (top_right_lon, top_right_lat)), gdf['proj:epsg'].values[0]
    
    return ((minx, miny), (maxx, maxy)), gdf['proj:epsg'].values[0]

def get_event_bbox(event_name, extra_mt = 0, when = None, return_proj_coords = False):
    
    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0

    crs_set = set()
    first_crs = None
    for mosaic_name in names.get_mosaics_names(event_name, when = when):
        ((tmp_minx, tmp_miny), (tmp_maxx, tmp_maxy)), crs = get_mosaic_bbox(event_name, mosaic_name, extra_mt = extra_mt, return_proj_coords = True)
        first_crs = crs if first_crs is None else first_crs
        transformer = pyproj.Transformer.from_crs(crs, first_crs)
        tmp_minx, tmp_miny = transformer.transform(tmp_minx, tmp_miny)
        tmp_maxx, tmp_maxy = transformer.transform(tmp_maxx, tmp_maxy)

        crs_set.add(crs)
        if tmp_minx < minx:
            minx = tmp_minx
        if tmp_miny < miny:
            miny = tmp_miny
        if tmp_maxx > maxx:
            maxx = tmp_maxx
        if tmp_maxy > maxy:
            maxy = tmp_maxy
    
    if not return_proj_coords:
        
        source_crs = first_crs #list(crs_set)[0]
        target_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        bott_left_lat, bott_left_lon = transformer.transform(minx, miny)
        top_right_lat, top_right_lon = transformer.transform(maxx, maxy)

        return (bott_left_lon, bott_left_lat), (top_right_lon, top_right_lat)
    
    return (minx, miny), (maxx, maxy)