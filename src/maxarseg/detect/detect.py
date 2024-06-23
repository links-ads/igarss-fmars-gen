from maxarseg.detect import detect_utils
from groundingdino.util.inference import predict as GD_predict
import numpy as np
from typing import List
import geopandas as gpd
from maxarseg.samplers import samplers, samplers_utils
from maxarseg.geo_datasets import geoDatasets
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import BoundingBox



def get_GD_boxes(img_batch: np.array, #b,h,w,c
                    GDINO_model,
                    TEXT_PROMPT,
                    BOX_THRESHOLD,
                    TEXT_THRESHOLD,
                    dataset_res,
                    device,
                    max_area_mt2 = 3000,
                    min_edges_ratio = 0,
                    reduce_perc = 0):
    
    batch_tree_boxes4Sam = []
    sample_size = img_batch.shape[1]
    num_trees4img = []

    for img in img_batch:
        image_transformed = detect_utils.GD_img_load(img)
        tree_boxes, logits, phrases = GD_predict(GDINO_model, image_transformed, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, device = device)
        #tree_boxes4Sam = []
        if len(tree_boxes) > 0:
            keep_ix_tree_boxes_area = detect_utils.filter_on_box_area_mt2(tree_boxes, sample_size, dataset_res, max_area_mt2 = max_area_mt2)
            keep_ix_tree_boxes_ratio = detect_utils.filter_on_box_ratio(tree_boxes, min_edges_ratio = min_edges_ratio)
            keep_ix_tree_boxes  = keep_ix_tree_boxes_area & keep_ix_tree_boxes_ratio
            
            reduced_tree_boxes = detect_utils.reduce_tree_boxes(tree_boxes[keep_ix_tree_boxes], reduce_perc = reduce_perc)
            
            tree_boxes4Sam = detect_utils.GDboxes2SamBoxes(reduced_tree_boxes, sample_size)
            
            num_trees4img.append(tree_boxes4Sam.shape[0])
            batch_tree_boxes4Sam.append(tree_boxes4Sam)
        else:
            num_trees4img.append(0)
            batch_tree_boxes4Sam.append(np.empty((0,4)))
    return batch_tree_boxes4Sam, np.array(num_trees4img)

"""def get_GD_boxes_tile_optimized(img_batch: np.array, #b,h,w,c
                                GDINO_model,
                                TEXT_PROMPT,
                                BOX_THRESHOLD,
                                TEXT_THRESHOLD,
                                dataset_res,
                                device,
                                max_area_mt2 = 3000,
                                min_edges_ratio = 0,
                                reduce_perc = 0):
    
    batch_tree_boxes4Sam = []
    sample_size = img_batch.shape[1]
    num_trees4img = []

    for img in img_batch:
        image_transformed = detect_utils.GD_img_load(img)
        tree_boxes, logits, phrases = GD_predict(GDINO_model, image_transformed, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, device = device)
        #tree_boxes4Sam = []
        if len(tree_boxes) > 0:
            keep_ix_tree_boxes_area = detect_utils.filter_on_box_area_mt2(tree_boxes, sample_size, dataset_res, max_area_mt2 = max_area_mt2)
            keep_ix_tree_boxes_ratio = detect_utils.filter_on_box_ratio(tree_boxes, min_edges_ratio = min_edges_ratio)
            keep_ix_tree_boxes  = keep_ix_tree_boxes_area & keep_ix_tree_boxes_ratio
            
            reduced_tree_boxes = detect_utils.reduce_tree_boxes(tree_boxes[keep_ix_tree_boxes], reduce_perc = reduce_perc)
            
            tree_boxes4Sam = detect_utils.GDboxes2SamBoxes(reduced_tree_boxes, sample_size)
            
            num_trees4img.append(tree_boxes4Sam.shape[0])
            batch_tree_boxes4Sam.append(tree_boxes4Sam)
        else:
            num_trees4img.append(0)
            batch_tree_boxes4Sam.append(np.empty((0,4)))
    return batch_tree_boxes4Sam, np.array(num_trees4img)"""
    
# OLD FUNCTION
def get_batch_buildings_boxes(batch_bbox: List[BoundingBox], proj_buildings_gdf: gpd.GeoDataFrame, dataset_res, ext_mt = 10):
    batch_building_boxes = []
    num_build4img = []
    for bbox in batch_bbox:
        query_bbox_poly = samplers_utils.boundingBox_2_Polygon(bbox) #from patch bbox to polygon
        index_MS_buildings = proj_buildings_gdf.sindex #get spatial index
        buildig_hits = index_MS_buildings.query(query_bbox_poly) #query buildinds index
        num_build4img.append(len(buildig_hits)) #append number of buildings
        #building_boxes = [] 
        if len(buildig_hits) > 0: #if there are buildings from proj geo cords to indexes
            building_boxes = samplers_utils.rel_bbox_coords(proj_buildings_gdf.iloc[buildig_hits], query_bbox_poly.bounds, dataset_res, ext_mt=ext_mt)
            building_boxes = np.array(building_boxes)
        else: #append empty array if no buildings
            building_boxes = np.empty((0,4))
        
        batch_building_boxes.append(building_boxes)

    return batch_building_boxes, np.array(num_build4img)

def get_batch_boxes(batch_bbox: List[BoundingBox], proj_gdf: gpd.GeoDataFrame, dataset_res, ext_mt = 0):
    """
    Given a batch of bounding boxes in a proj crs, it returns the boxes in the right coordinates relative to the sampled patch. 
    It is necessary that the bbox and the gdf are in the same crs.
    """
    batch_boxes = []
    num_boxes4img = []
    gdf_index = proj_gdf.sindex
    for bbox in batch_bbox:
        query_patch_poly = samplers_utils.boundingBox_2_Polygon(bbox) #from patch bbox to polygon
    
        hits = gdf_index.query(query_patch_poly) #query index
        
        num_boxes4img.append(len(hits)) #append number of boxes
        
        if len(hits) > 0: #if there is at least a box in the query_bbox_poly
            
            boxes = samplers_utils.rel_bbox_coords(geodf = proj_gdf.iloc[hits],
                                                    ref_coords = query_patch_poly.bounds,
                                                    res = dataset_res,
                                                    ext_mt = ext_mt)
            boxes = np.array(boxes)
        else: #append empty array if no buildings
            boxes = np.empty((0,4))
        
        batch_boxes.append(boxes)
    
    return batch_boxes, np.array(num_boxes4img)

def get_refined_batch_boxes(batch_bbox: List[BoundingBox], proj_gdf: gpd.GeoDataFrame, dataset_res, ext_mt = 0):
    if len(batch_bbox) != 1:
        raise ValueError("Invalid input: batch_bbox should contain exactly one bounding box (segmentation batch size must be 1)")
    bbox = batch_bbox[0]    
    query_patch_poly = samplers_utils.boundingBox_2_Polygon(bbox) #from patch bbox to polygon
    try:
        intersec_geom = proj_gdf.intersection(query_patch_poly)
    except Exception as e:
        proj_gdf['geometry'] = proj_gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
        intersec_geom = proj_gdf.intersection(query_patch_poly)
    valid_gdf = intersec_geom[~intersec_geom.is_empty]
    num_boxes4img = [len(valid_gdf)]
    if len(valid_gdf) > 0:
        boxes = samplers_utils.rel_bbox_coords(geodf = valid_gdf,
                                                ref_coords = query_patch_poly.bounds,
                                                res = dataset_res,
                                                ext_mt = ext_mt)
    else:
        boxes = np.empty((0,4))
    
    batch_boxes = [boxes]
    
    return batch_boxes, np.array(num_boxes4img)