import rasterio
from pathlib import Path
import numpy as np
from maxarseg.ESAM_segment import segment_utils
import pandas as pd
from maxarseg.polygonize import polygonize_with_values
import geopandas as gpd

def single_mask2Tif(tile_path, mask, out_name, out_dir_root = './output/tiff'):
    """
    Converts a binary mask to a GeoTIFF file.

    Args:
        tile_path (str): The path to the input tile file.
        mask (numpy.ndarray): The binary mask array.
        out_name (str): The name of the output GeoTIFF file.
        out_dir_root (str, optional): The root directory for the output GeoTIFF file. Defaults to './output/tiff'.

    Returns:
        None
    """
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1})
    out_path = Path(out_dir_root) / out_name
    
    with rasterio.open(out_path, 'w', **out_meta) as dest:
            dest.write(mask, 1) 
    
    print(f"Mask written in {out_path}")
    
def masks2Tifs(tile_path , masks: np.ndarray, out_names: list, separate_masks: bool, out_dir_root = './output/tiff'):
    if not separate_masks: #merge the masks
        mask = segment_utils.merge_masks(masks)
        masks = np.expand_dims(mask, axis=0)
    
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1})
    masks = masks.astype(np.uint8)
    for j, out_name in enumerate(out_names):
        out_path = Path(out_dir_root) / out_name
        with rasterio.open(out_path, 'w', **out_meta) as dest:
                dest.write(masks[j], 1)   
        print(f"Mask written in {out_path}")
    
    return masks
    
def gen_names(tile_path, separate_masks=False):
    """
    Generate output file names based on the given tile path.

    Args:
        tile_path (Path): The path to the tile file.
        divide_masks (bool, optional): Whether to divide masks into separate files. Defaults to False.

    Returns:
        list: A list of output file names.
    """
    ev_name, tl_when, mos_name, tl_name = tile_path.parts[-4:]
    masks_names = ['road', 'tree', 'building']
    
    if separate_masks:
        out_names = [Path(ev_name) / tl_when / mos_name / (tl_name.split('.')[0] + '_' + mask_name + '.tif') for mask_name in masks_names]        
    else:
        out_names = [Path(ev_name) / tl_when / mos_name / (tl_name.split('.')[0] + '.tif')]
    
    return out_names

def masks2parquet(tile_path , tree_build_masks: np.ndarray, road_series: pd.Series, out_names: list, out_dir_root = './output/tiff'):
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
        # convert no_overlap_masks to int
    tolerances = [0.001, 0.005]
    pixel_thresholds = [20, 20]
    # polygonization
    with rasterio.open(tile_path) as src:
        out_meta = src.meta.copy()
    gdf_list = [] 
    # convert pd.Series to gpd.GeoDataFrame
    road_gdf = gpd.GeoDataFrame(road_series)
    # set road_gdf class_id to 0
    road_gdf['class_id'] = 0
    # rename the columns
    road_gdf.columns = ['geometry', 'class_id']
    gdf_list.append(road_gdf)
    # cicling over the masks channels
    for i in range(tree_build_masks.shape[0]):
        if tree_build_masks[i].sum() != 0:
            gdf = polygonize_with_values(tree_build_masks[i], class_id=i+1, tolerance=tolerances[i], transform=out_meta['transform'], crs=out_meta['crs'], pixel_threshold=pixel_thresholds[i])
            gdf_list.append(gdf)
    crs = out_meta['crs']
    # Set the CRS of all GeoDataFrames to the same CRS
    for gdf in gdf_list:
        gdf.set_geometry('geometry', inplace=True)
        gdf.crs = crs
    gdf_list = [gdf.to_crs(crs) for gdf in gdf_list if 'geometry' in gdf.columns and gdf['geometry'].notna().all()]
    # concatenate out_dir_root with out_names[0]
    out_path = out_dir_root / out_names[0]
    # replace '.tif' with '.parquet'
    out_path = out_path.with_suffix('.parquet')
    # create a single gdf
    gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    print('Parquet file created at:', out_path)
    # create gdf_first with the first row of gdf
    assert out_names.__len__() == 1, "Only one output name is allowed for parquet file"
    gdf.to_parquet(out_path)
    return gdf