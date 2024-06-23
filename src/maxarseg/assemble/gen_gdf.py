import pandas as pd
from pathlib import Path
import geopandas as gpd
from pyquadkey2 import quadkey
import pandas as pd
from shapely import geometry
import json
from maxarseg import assemble
import time
from shapely.wkt import loads


def intersecting_qks(bott_left_lon_lat: tuple, top_right_lon_lat: tuple, min_level=7, max_level=9):
    """
    Get the quadkeys that intersect the given bbox.
    Input:
        bott_left_lon_lat: Tuple with the coordinates of the bottom left corner. Example: (-16.5, 13.5)
        top_right_lon_lat: Tuple with the coordinates of the top right corner. Example: (-15.5, 14.5)
        min_level: Minimum level of the quadkeys
        max_level: Maximum level of the quadkeys
    """
    bott_left_lon, bott_left_lat = bott_left_lon_lat
    top_right_lon, top_right_lat = top_right_lon_lat
    qk_bott_left = quadkey.from_geo((bott_left_lat, bott_left_lon), level=max_level) #lat, lon
    qk_top_right = quadkey.from_geo((top_right_lat, top_right_lon), level=max_level) #lat, lon
    hits = qk_bott_left.difference(qk_top_right)
    candidate_hits = set()

    for hit in hits:
        current_qk = hit
        for _ in range(min_level, max_level):
            current_qk = current_qk.parent()
            candidate_hits.add(current_qk)
    hits.extend(candidate_hits)
    return [int(str(hit)) for hit in hits]

def qk_building_gdf(qk_list, csv_path = 'metadata/buildings_dataset_links.csv', dataset_crs = None, quiet = False):
    """
    Returns a geodataframe with the buildings in the quadkeys given as input.
    It downloads the dataset from a link in the dataset-links.csv file.
    Coordinates are converted in the crs passed as input.

    Inputs:
        qk_list: the list of quadkeys to look for in the csv
        root: the root directory of the dataset-links.csv file
        dataset_crs: the crs in which to convert the coordinates of the buildings
        quiet: if True, it doesn't print anything
    """
    building_links_df = pd.read_csv(csv_path)
    country_links = building_links_df[building_links_df['QuadKey'].isin(qk_list)]

    if not quiet:
        print(f"\nBuildings: found {len(country_links)} links matching: {qk_list}")

    if len(country_links) == 0:
        print("MS-Buildings: No buildings for this region")
        return gpd.GeoDataFrame()
    gdfs = []
    for _, row in country_links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(geometry.shape)
        gdf_down = gpd.GeoDataFrame(df, crs=4326)
        gdfs.append(gdf_down)

    gdfs = pd.concat(gdfs)
    if dataset_crs is not None: #if the crs is passed, convert the coordinates
        gdfs = gdfs.to_crs(dataset_crs)
    return gdfs

def google_building_gdf(event_name, bbox):
    """
    Generate a GeoDataFrame containing Google building data filtered by a bounding box.

    Args:
    - event_name (str): The name of the event.
    - bbox (tuple): A tuple representing the bounding box coordinates in the format (minx, miny, maxx, maxy).

    Returns:
    - gdf_filtered (GeoDataFrame): A GeoDataFrame containing the filtered Google building data.
    """ 
    root = '/nfs/projects/overwatch/maxar-segmentation/google-open-buildings'
    f_name = 'open_buildings_v3_'+ event_name + '.csv'
    file_path = Path(root) / f_name
    df = pd.read_csv(file_path) #TODO: leggere solo le colonne necessarie e filtrare magari su confidence boxes
    df['geometry'] = df['geometry'].apply(loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    gdf_filtered = assemble.filter.filter_gdf_w_bbox(gdf, bbox)
    return gdf_filtered

def get_region_road_gdf(region_name, roads_root = '/nfs/projects/overwatch/maxar-segmentation/microsoft-roads'):
    #TODO: cercare di velocizzare la lettura dei dati delle strade
    """
    Get a gdf containing the roads of a region.
    Input:
        region_name: Name of the region. Example: 'AfricaWest-Full'
        roads_root: Root directory of the roads datasets
    """
    start_time = time.time()
    print(f'Roads: reading roads for the whole {region_name} region')
    if region_name[-4:] != '.tsv':
        region_name = region_name + '.tsv'
    
    def custom_json_loads(s):
        try:
            return geometry.shape(json.loads(s)['geometry'])
        except:
            return geometry.LineString()

    roads_root = Path(roads_root)
    if region_name != 'USA.tsv':
        print('Roads: not in USA. Region name:', region_name)
        region_road_df = pd.read_csv(roads_root/region_name, names =['country', 'geometry'], sep='\t')
    else:
        print('is USA:', region_name)
        region_road_df = pd.read_csv(roads_root/region_name, names =['geometry'], sep='\t')
    #region_road_df['geometry'] = region_road_df['geometry'].apply(json.loads).apply(lambda d: geometry.shape(d.get('geometry')))
    #slightly faster
    region_road_df['geometry'] = region_road_df['geometry'].apply(custom_json_loads)
    region_road_gdf = gpd.GeoDataFrame(region_road_df, crs=4326)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for reading roads: {elapsed_time:.2f} seconds")
    return region_road_gdf