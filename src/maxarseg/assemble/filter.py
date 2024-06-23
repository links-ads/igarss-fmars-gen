from shapely import geometry
import geopandas as gpd
from typing import List, Tuple, Union

def filter_gdf_w_bbox(gbl_gdf: gpd.GeoDataFrame, bbox: Union[List[Tuple], Tuple[Tuple]]) -> gpd.GeoDataFrame:
    """
    Filter a geodataframe with a bbox.
    Input:
        gbl_gdf: the geodataframe to be filtered
        mosaic_bbox: Bounding box of the mosaic in format (lon, lat). Example: ((-16.5, 13.5), (-15.5, 14.5))
    Output:
        a filtered geodataframe
    
    """
    (minx, miny), (maxx, maxy) = bbox
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)] #lon lat
    query_bbox_poly = geometry.Polygon(vertices)
    
    hits = gbl_gdf.geometry.intersects(query_bbox_poly)
    #TODO: magari funziona anche...
    #hits = gbl_gdf.sindex.query(query_bbox_poly)
    
    return gbl_gdf[hits]

def maybe_faster_filter_gdf_w_bbox(gbl_gdf: gpd.GeoDataFrame, bbox: Union[List[Tuple], Tuple[Tuple]]) -> gpd.GeoDataFrame:
    """
    Filter a geodataframe with a bbox.
    Input:
        gbl_gdf: the geodataframe to be filtered
        mosaic_bbox: Bounding box of the mosaic in format (lon, lat). Example: ((-16.5, 13.5), (-15.5, 14.5))
    Output:
        a filtered geodataframe
    
    """
    (minx, miny), (maxx, maxy) = bbox
    vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)] #lon lat
    query_bbox_poly = geometry.Polygon(vertices)
    
    hits = gbl_gdf.sindex.query(query_bbox_poly)
    
    return gbl_gdf[hits]