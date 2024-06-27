#Generic
from pathlib import Path
from tqdm import tqdm
import threading

import os
import sys
from time import time, perf_counter
import numpy as np
import rasterio
from rasterio.features import rasterize
import warnings
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
import torchvision
import geopandas as gpd
from typing import Tuple
import matplotlib.pyplot as plt
from shapely import geometry
import json
import pandas as pd

#My functions
from maxarseg.assemble import delimiters, filter, gen_gdf, names
from maxarseg.ESAM_segment import segment, segment_utils
from maxarseg.samplers import samplers, samplers_utils
from maxarseg.geo_datasets import geoDatasets
from maxarseg.configs import SegmentConfig, DetectConfig
from maxarseg.detect import detect, detect_utils
from maxarseg import output
from maxarseg import plotting_utils

#GroundingDino
from groundingdino.util.inference import load_model as GD_load_model
from groundingdino.util.inference import predict as GD_predict

#Deep forest
from deepforest import main

#esam
from maxarseg.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

# Ignore all warnings
warnings.filterwarnings('ignore')


class Mosaic:
    def __init__(self,
                 name,
                 event
                 ):
        
        #Mosaic
        self.name = name
        self.event = event
        self.bbox, self.crs = delimiters.get_mosaic_bbox(self.event.name,
                                          self.name,
                                          self.event.maxar_metadata_path,
                                          extra_mt=1000)
        
        self.when = list((self.event.maxar_root / self.event.name).glob('**/*'+self.name))[0].parts[-2]
        self.tiles_paths = list((self.event.maxar_root / self.event.name / self.when / self.name).glob('*.tif'))
        self.tiles_num = len(self.tiles_paths)
        
        #Check if img is bw
        with rasterio.open(self.tiles_paths[0]) as src:
            num_bands = src.count
        if num_bands == 1:
            print(f'Image {self.tiles_paths[0]} is in black and white', )
            self.is_rgb = False
        else:
            self.is_rgb = True

        #Roads
        self.road_gdf = None
        self.proj_road_gdf = None
        self.road_num = None

        #Buildings
        self.build_gdf = None
        self.proj_build_gdf = None
        self.sindex_proj_build_gdf = None
        self.build_num = None

        #models
        self.GD_model = None
        self.DF_model = None
        self.ESAM_model = None

    def __str__(self) -> str:
        return self.name
    
    def set_road_gdf(self):
        if self.event.road_gdf is None:
            self.event.set_road_gdf()

        self.road_gdf = filter.filter_gdf_w_bbox(self.event.road_gdf, self.bbox)
        self.proj_road_gdf =  self.road_gdf.to_crs(self.crs)
        self.road_num = len(self.road_gdf)
        print(f'Roads in {self.name} mosaic: {self.road_num}')
    
    def set_build_gdf(self):
        qk_hits = gen_gdf.intersecting_qks(*self.bbox)
        self.build_gdf = gen_gdf.qk_building_gdf(qk_hits, csv_path = self.event.buildings_ds_links_path)

        if len(self.build_gdf) == 0: #here use google buildings
            self.build_gdf = None
            self.proj_build_gdf = None
            print('No buildings found for this mosaic')
            return False
                
        self.proj_build_gdf = self.build_gdf.to_crs(self.crs)
        self.sindex_proj_build_gdf = self.proj_build_gdf.sindex
    
    # Method no more used  
    def seg_road_tile(self, tile_path, aoi_mask) -> np.ndarray:
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_h = src.height
            tile_w = src.width
            tile_shape = (tile_h, tile_w)
            
        cfg = self.event.cfg
        #aoi_mask = rasterize(tile_aoi_gdf.geometry, out_shape = tile_shape, fill=False, default_value=True, transform = transform)

        query_bbox_poly = samplers_utils.path_2_tile_aoi(tile_path)
        road_lines = self.proj_road_gdf[self.proj_road_gdf.geometry.intersects(query_bbox_poly)]

        if len(road_lines) != 0:
            buffered_lines = road_lines.geometry.buffer(cfg.get('segmentation/roads/road_width_mt'))
            road_mask = rasterize(buffered_lines, out_shape=(tile_h, tile_w), transform=transform)
            road_mask = np.where(aoi_mask, road_mask, False)
        else:
            print('No roads')
            road_mask = np.zeros((tile_h, tile_w))
        return road_mask  #shape: (h, w)
    
    def polyg_road_tile(self, tile_aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
        cfg = self.event.cfg
        road_lines = samplers_utils.filter_road_gdf_vs_aois_gdf(self.proj_road_gdf, tile_aoi_gdf)
        if len(road_lines) != 0:
            buffered_lines = road_lines.geometry.buffer(cfg.get('segmentation/roads/road_width_mt'))
            intersected_buffered_lines_ser = samplers_utils.intersection_road_gdf_vs_aois_gdf(buffered_lines, tile_aoi_gdf)
        else :
            print('No roads')
            intersected_buffered_lines_ser = gpd.GeoSeries()
        return intersected_buffered_lines_ser
    
    def detect_trees_tile_DeepForest(self, tile_path) -> Tuple[np.ndarray, ...]:
        cfg = self.event.cfg
        if self.DF_model is None:
            self.DF_model = main.deepforest(config_args = { 'devices' : cfg.get('models/df/device'),
                                                    'retinanet': {'score_thresh': cfg.get('models/df/box_threshold')},
                                                    'accelerator': 'cuda',
                                                    'batch_size': cfg.get('models/df/bs')})
            self.DF_model.use_release()
        
        boxes_df = self.DF_model.predict_tile(tile_path,
                                    return_plot = False,
                                    patch_size = cfg.get('models/df/size'),
                                    patch_overlap = cfg.get('models/df/patch_overlap'))
        
        
        boxes = boxes_df.iloc[:, :4].values
        score = boxes_df['score'].values
        
        return boxes, score
    
    def noTGeo_detect_trees_tile_GD(self, tile_path, tile_aoi_gdf: gpd.GeoDataFrame, aoi_mask) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.event.cfg
        #load model
        model = GD_load_model(cfg.get('models/gd/config_file_path'), cfg.get('models/gd/weight_path')).to(cfg.get('models/gd/device'))
        print('\n- GD model device:', next(model.parameters()).device)
        
        dataset = geoDatasets.SingleTileDataset(str(tile_path), tile_aoi_gdf, aoi_mask)
        sampler = samplers.SinglePatchSampler(dataset, patch_size=cfg.get('models/esam/size'), stride=cfg.get('models/esam/stride'))
        dataloader = DataLoader(dataset, sampler=sampler, collate_fn=geoDatasets.single_sample_collate_fn)
        
        glb_tile_tree_boxes = torch.empty(0, 4)
        all_logits = torch.empty(0)
        
        for batch in tqdm(dataloader, total = len(dataloader), desc="Detecting Trees with GDino"):
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8')
            
            for img, img_top_left_index in zip(img_b, batch['top_lft_index']):
                image_transformed = detect_utils.GD_img_load(img)
                tree_boxes, logits, phrases = GD_predict(model,
                                                         image_transformed,
                                                         cfg.get('models/gd/text_prompt'),
                                                         cfg.get('models/gd/box_threshold'),
                                                         cfg.get('models/gd/text_threshold'),
                                                         device = cfg.get('models/gd/device'))
                
                rel_xyxy_tree_boxes = detect_utils.GDboxes2SamBoxes(tree_boxes, img_shape = cfg.get('models/gd/size'))
                top_left_xy = np.array([img_top_left_index[1], #from an index to xyxy
                                        img_top_left_index[0],
                                        img_top_left_index[1],
                                        img_top_left_index[0]])
                
                #turn boxes from patch xyxy coords to global xyxy coords
                glb_xyxy_tree_boxes = rel_xyxy_tree_boxes + top_left_xy
                
                glb_tile_tree_boxes = np.concatenate((glb_tile_tree_boxes, glb_xyxy_tree_boxes))
                all_logits = np.concatenate((all_logits, logits))
        
        #del model and free GPU
        #TODO: if enough space in GPU, keep the model loaded
        del model
        
        return glb_tile_tree_boxes, all_logits  
    
    def detect_trees_tile_GD(self, tile_path, tile_aoi_gdf: gpd.GeoDataFrame, aoi_mask) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.event.cfg
        #load model
        model = GD_load_model(cfg.get('models/gd/config_file_path'), cfg.get('models/gd/weight_path')).to(cfg.get('models/gd/device'))
        print('\n- GD model device:', next(model.parameters()).device)
        
        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path), tile_aoi_gdf, aoi_mask=aoi_mask)
        sampler = samplers.BatchGridGeoSampler(dataset, batch_size=cfg.get('models/gd/bs'), size=cfg.get('models/gd/size'), stride=cfg.get('models/gd/stride'))
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)
        
        glb_tile_tree_boxes = torch.empty(0, 4)
        all_logits = torch.empty(0)
        
        for batch in tqdm(dataloader, total = len(dataloader), desc="Detecting Trees with GDino"):
            img_b = batch['image'].permute(0,2,3,1).numpy().astype('uint8')
            
            for img, img_top_left_index in zip(img_b, batch['top_lft_index']):
                image_transformed = detect_utils.GD_img_load(img)
                tree_boxes, logits, phrases = GD_predict(model,
                                                         image_transformed,
                                                         cfg.get('models/gd/text_prompt'),
                                                         cfg.get('models/gd/box_threshold'),
                                                         cfg.get('models/gd/text_threshold'),
                                                         device = cfg.get('models/gd/device'))
                
                rel_xyxy_tree_boxes = detect_utils.GDboxes2SamBoxes(tree_boxes, img_shape = cfg.get('models/gd/size'))
                top_left_xy = np.array([img_top_left_index[1], #from an index to xyxy
                                        img_top_left_index[0],
                                        img_top_left_index[1],
                                        img_top_left_index[0]])
                
                #turn boxes from patch xyxy coords to global xyxy coords
                glb_xyxy_tree_boxes = rel_xyxy_tree_boxes + top_left_xy
                
                glb_tile_tree_boxes = np.concatenate((glb_tile_tree_boxes, glb_xyxy_tree_boxes))
                all_logits = np.concatenate((all_logits, logits))
        
        #del model and free GPU
        #TODO: if enough space in GPU, keep the model loaded
        del model
        
        return glb_tile_tree_boxes, all_logits        
    
    def detect_trees_tile(self, tile_path, tile_aoi_gdf, aoi_mask, georef = True):
        with rasterio.open(tile_path) as src:
            to_xy = src.xy
            crs = src.crs
            
        cfg = self.event.cfg
        if cfg.get('detection/trees/use_DF'):
            deepForest_glb_tile_tree_boxes, deepForest_scores = self.detect_trees_tile_DeepForest(tile_path)
        if cfg.get('detection/trees/use_GD'):
            GD_glb_tile_tree_boxes, GD_scores = self.noTGeo_detect_trees_tile_GD(tile_path, tile_aoi_gdf, aoi_mask)
        
        if cfg.get('detection/trees/use_DF') and cfg.get('detection/trees/use_GD'):
            glb_tile_tree_boxes = np.concatenate((GD_glb_tile_tree_boxes, deepForest_glb_tile_tree_boxes))
            glb_tile_tree_scores = np.concatenate((GD_scores, deepForest_scores))
        elif cfg.get('detection/trees/use_DF'):
            glb_tile_tree_boxes = deepForest_glb_tile_tree_boxes
            glb_tile_tree_scores = deepForest_scores
        elif cfg.get('detection/trees/use_GD'):
            glb_tile_tree_boxes = GD_glb_tile_tree_boxes
            glb_tile_tree_scores = GD_scores
        
        print('Number of tree boxes before filtering: ', len(glb_tile_tree_boxes))
                
        keep_ix_box_area = detect_utils.filter_on_box_area_mt2(glb_tile_tree_boxes,
                                                               max_area_mt2 = cfg.get('detection/trees/max_area_boxes_mt2'),
                                                               box_format = 'xyxy')
        glb_tile_tree_boxes = glb_tile_tree_boxes[keep_ix_box_area]
        glb_tile_tree_scores = glb_tile_tree_scores[keep_ix_box_area]
        print('boxes area filtering: ', len(keep_ix_box_area) - np.sum(keep_ix_box_area), 'boxes removed')
        
        keep_ix_box_ratio = detect_utils.filter_on_box_ratio(glb_tile_tree_boxes,
                                                             min_edges_ratio = cfg.get('detection/trees/min_ratio_GD_boxes_edges'),
                                                             box_format = 'xyxy')
        glb_tile_tree_boxes = glb_tile_tree_boxes[keep_ix_box_ratio]
        glb_tile_tree_scores = glb_tile_tree_scores[keep_ix_box_ratio]
        print('box edge ratio filtering:', len(keep_ix_box_ratio) - np.sum(keep_ix_box_ratio), 'boxes removed')
        
        keep_ix_nms = torchvision.ops.nms(torch.tensor(glb_tile_tree_boxes), torch.tensor(glb_tile_tree_scores.astype(np.float64)), cfg.get('detection/trees/nms_threshold'))
        len_bf_nms = len(glb_tile_tree_boxes)
        glb_tile_tree_boxes = glb_tile_tree_boxes[keep_ix_nms]
        glb_tile_tree_scores = glb_tile_tree_scores[keep_ix_nms]
        print('nms filtering:', len_bf_nms - len(keep_ix_nms), 'boxes removed')
        
        if len(glb_tile_tree_boxes.shape) == 1:
            glb_tile_tree_boxes = np.expand_dims(glb_tile_tree_boxes, axis = 0)
        if glb_tile_tree_scores.size == 1:
            glb_tile_tree_scores = np.expand_dims(glb_tile_tree_scores, axis = 0)
        
        if georef: #create a gdf with the boxes in proj coordinates
            for i, box in enumerate(glb_tile_tree_boxes):
                #need to invert x and y to go from col row to row col index
                try:
                    glb_tile_tree_boxes[i] = np.array(to_xy(box[1], box[0]) + to_xy(box[3], box[2]))
                # catch and print 
                except Exception as e:
                    print(f'Error in box {i}: {e}')                
            cols = {'score': list(glb_tile_tree_scores),
                    'geometry': [samplers_utils.xyxyBox2Polygon(box) for box in glb_tile_tree_boxes]}
            
            gdf = gpd.GeoDataFrame(cols, crs = crs)
            
            if self.event.cross_wlb == True:
                #keep only tree detections that are inside tile_aoi_gdf
                gdf = gdf[gdf['geometry'].apply(lambda x: tile_aoi_gdf.intersects(x).any())]
                print("Not in aoi:", len(glb_tile_tree_scores) - len(gdf), "boxes removed")
                print('Number of tree boxes after all filtering: ', len(gdf))
            return gdf
            
        return glb_tile_tree_boxes #xyxy format, global (tile) index
    
    def noTGeo_seg_glb_tree_and_build_tile(self, tile_path: str, tile_aoi_gdf: gpd.GeoDataFrame, aoi_mask: np.ndarray):
        cfg = self.event.cfg
        if self.build_gdf is None: #set buildings at mosaic level
            self.set_build_gdf()
        
        tile_building_gdf = self.proj_build_gdf.iloc[self.sindex_proj_build_gdf.query(samplers_utils.path_2_tile_aoi(tile_path))]
        
        trees_gdf = self.detect_trees_tile(tile_path, tile_aoi_gdf = tile_aoi_gdf, aoi_mask = aoi_mask, georef = True)
        
        dataset = geoDatasets.SingleTileDataset(str(tile_path), tile_aoi_gdf, aoi_mask)
        sampler = samplers.SinglePatchSampler(dataset, patch_size=cfg.get('models/esam/size'), stride=cfg.get('models/esam/stride'))
        dataloader = DataLoader(dataset, sampler=sampler, collate_fn=geoDatasets.single_sample_collate_fn)
        
        canvas = np.zeros((2,) + dataset.tile_shape, dtype=np.float32) # dim (3, h_tile, w_tile). The dim 0 is: tree, build
        weights = np.zeros(dataset.tile_shape, dtype=np.float32) # dim (h_tile, w_tile)
        for _, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc = "Segmenting"):
            original_img_tsr = batch['image']

            #TREES
            #get the tree boxes in batches and the number of trees for each image
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            if len(trees_gdf) == 0:
                tree_boxes_b = [np.empty((0, 4))]
                num_trees4img = [0]
            else:
                tree_boxes_b, num_trees4img = detect.get_batch_boxes(batch['bbox'],
                                                                    proj_gdf = trees_gdf,
                                                                    dataset_res = dataset.res,
                                                                    ext_mt = 0)
            
            #BUILDINGS
            #get the building boxes in batches and the number of buildings for each image
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            building_boxes_b, num_build4img = detect.get_refined_batch_boxes(batch['bbox'],
                                                                    proj_gdf = tile_building_gdf,
                                                                    dataset_res = dataset.res,
                                                                    ext_mt = cfg.get('detection/buildings/ext_mt_build_box'))

            if num_trees4img[0] > 0 or num_build4img[0] > 0:
                
                max_detect = max(num_trees4img + num_build4img)
                
                #obtain the right input for the ESAM model (trees + buildings)
                input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
                
                # segment the image and get for each image as many masks as the number of boxes,
                # for GPU constraint use num_parall_queries
                tree_build_mask = segment.ESAM_from_inputs_fast(original_img_tsr = original_img_tsr,
                                                            input_points = torch.from_numpy(input_points),
                                                            input_labels = torch.from_numpy(input_labels),
                                                            num_tree_boxes= num_trees4img,
                                                            efficient_sam = self.event.efficient_sam,
                                                            device = cfg.get('models/esam/device'),
                                                            num_parall_queries = cfg.get('models/esam/num_parall_queries'))
            
            else:
                #print('no prompts in patch, skipping...')
                tree_build_mask = np.zeros((2, *original_img_tsr.shape[2:]), dtype = np.float32) #(2, h, w)
            
            canvas, weights = segment_utils.write_canvas_geo_window(canvas = canvas,
                                                                    weights = weights,
                                                                    patch_masks_b = np.expand_dims(tree_build_mask, axis=0),
                                                                    top_lft_indexes = batch['top_lft_index'],
                                                                    )

        canvas = np.divide(canvas, weights, out=np.zeros_like(canvas), where=weights!=0)
        canvas = np.greater(canvas, 0) #turn logits into bool
        canvas = np.where(aoi_mask, canvas, False)
        return canvas
    
    def seg_glb_tree_and_build_tile_fast(self, tile_path: str, tile_aoi_gdf: gpd.GeoDataFrame, aoi_mask: np.ndarray):
        cfg = self.event.cfg
        if self.build_gdf is None: #set buildings at mosaic level
            self.set_build_gdf()
        
        tile_building_gdf = self.proj_build_gdf.iloc[self.sindex_proj_build_gdf.query(samplers_utils.path_2_tile_aoi(tile_path))]
        
        trees_gdf = self.detect_trees_tile(tile_path, tile_aoi_gdf = tile_aoi_gdf, aoi_mask = aoi_mask, georef = True)
        
        dataset = geoDatasets.MxrSingleTileNoEmpty(str(tile_path), tile_aoi_gdf, aoi_mask)
        sampler = samplers.BatchGridGeoSampler(dataset,
                                            batch_size=cfg.get('models/esam/bs'),
                                            size=cfg.get('models/esam/size'),
                                            stride=cfg.get('models/esam/stride'))
        dataloader = DataLoader(dataset , batch_sampler=sampler, collate_fn=stack_samples)
        
        canvas = np.zeros((2,) + samplers_utils.tile_sizes(dataset), dtype=np.float32) # dim (3, h_tile, w_tile). The dim 0 is: tree, build
        weights = np.zeros(samplers_utils.tile_sizes(dataset), dtype=np.float32) # dim (h_tile, w_tile)
        for batch_ix, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc = "Segmenting"):
            original_img_tsr = batch['image']

            #TREES
            #get the tree boxes in batches and the number of trees for each image
            #tree_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di tree boxes
            if len(trees_gdf) == 0:
                tree_boxes_b = [np.empty((0, 4))]
                num_trees4img = [0]
            else:
                tree_boxes_b, num_trees4img = detect.get_batch_boxes(batch['bbox'],
                                                                    proj_gdf = trees_gdf,
                                                                    dataset_res = dataset.res,
                                                                    ext_mt = 0)
            
            #BUILDINGS
            #get the building boxes in batches and the number of buildings for each image
            #building_boxes_b è una lista con degli array di shape (n, 4) dove n è il numero di building boxes
            building_boxes_b, num_build4img = detect.get_refined_batch_boxes(batch['bbox'],
                                                                    proj_gdf = tile_building_gdf,
                                                                    dataset_res = dataset.res,
                                                                    ext_mt = cfg.get('detection/buildings/ext_mt_build_box'))

            if num_trees4img[0] > 0 or num_build4img[0] > 0:
                
                max_detect = max(num_trees4img + num_build4img)
                
                #obtain the right input for the ESAM model (trees + buildings)
                input_points, input_labels = segment_utils.get_input_pts_and_lbs(tree_boxes_b, building_boxes_b, max_detect)
                
                # segment the image and get for each image as many masks as the number of boxes,
                # for GPU constraint use num_parall_queries
                tree_build_mask = segment.ESAM_from_inputs_fast(original_img_tsr = original_img_tsr,
                                                            input_points = torch.from_numpy(input_points),
                                                            input_labels = torch.from_numpy(input_labels),
                                                            num_tree_boxes= num_trees4img,
                                                            efficient_sam = self.event.efficient_sam,
                                                            device = cfg.get('models/esam/device'),
                                                            num_parall_queries = cfg.get('models/esam/num_parall_queries'))
            
            else:
                #print('no prompts in patch, skipping...')
                tree_build_mask = np.zeros((2, *original_img_tsr.shape[2:]), dtype = np.float32) #(2, h, w)
            
            canvas, weights = segment_utils.write_canvas_geo_window(canvas = canvas,
                                                                    weights = weights,
                                                                    patch_masks_b = np.expand_dims(tree_build_mask, axis=0),
                                                                    top_lft_indexes = batch['top_lft_index'],
                                                                    )

        canvas = np.divide(canvas, weights, out=np.zeros_like(canvas), where=weights!=0)
        canvas = np.greater(canvas, 0) #turn logits into bool
        canvas = np.where(aoi_mask, canvas, False)
        return canvas
    
    def segment_tile(self, tile_path, out_dir_root, overwrite = False, separate_masks = True):
        """
        glbl_det: if True tree detection are computed at tile level, if False at patch level
        """
        
        #create folder if it does not exists
        tile_path = Path(tile_path)
        out_dir_root = Path(out_dir_root)
        out_names = output.gen_names(tile_path, separate_masks)
        (out_dir_root / out_names[0]).parent.mkdir(parents=True, exist_ok=True) #create folder if not exists
        if not overwrite:
            for out_name in out_names:
                if (out_dir_root / out_name).exists():
                    print(f'File {out_dir_root / out_name} already exists')
                    return True
                #assert not (out_dir_root / out_name).exists(), f'File {out_dir_root / out_name} already exists'
            
        tile_aoi_gdf = samplers_utils.path_2_tile_aoi_no_water(tile_path, self.event.filtered_wlb_gdf)
        
        if tile_aoi_gdf.iloc[0].geometry.is_empty: #tile completely on water
            pass
            # print("\nSave an empty mask")
            # thread = threading.Thread(target=self.save_all_blank,
            #                             args=(out_dir_root, tile_path, out_names, separate_masks))
            #self.save_all_blank(out_dir_root, tile_path, out_names, separate_masks)
        
        else:
            #retrieve roads and build at mosaic level if not already done
            if self.build_gdf is None:
                response = self.set_build_gdf()
                if response == False:
                    return False
            if self.road_gdf is None:
                self.set_road_gdf()
            with rasterio.open(tile_path) as src:
                transform = src.transform
                tile_shape = (src.height, src.width)
            aoi_mask = rasterize(tile_aoi_gdf.geometry, out_shape = tile_shape, fill=False, default_value=True, transform = transform)
            
            #tree_and_build_mask = self.seg_glb_tree_and_build_tile_fast(tile_path, tile_aoi_gdf, aoi_mask)            
            tree_and_build_mask = self.noTGeo_seg_glb_tree_and_build_tile(tile_path, tile_aoi_gdf, aoi_mask)
            
            #thread = threading.Thread(target=self.postprocess_and_save,
            #                            args=(tree_and_build_mask, out_dir_root, tile_path, out_names, tile_aoi_gdf, aoi_mask,separate_masks))
        
            #thread.start()
            self.postprocess_and_save(tree_and_build_mask, out_dir_root, tile_path, out_names, tile_aoi_gdf, aoi_mask, separate_masks)
        
        return True
    #TODO: not working but should be faster
    def seg_and_poly_road_tile(self, tile_path, tile_aoi_gdf):
        cfg = self.event.cfg
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_h = src.height
            tile_w = src.width
        
        intersected_buffered_lines_ser = self.polyg_road_tile(tile_aoi_gdf)
        if len(intersected_buffered_lines_ser) != 0:
            road_mask = rasterize(intersected_buffered_lines_ser, out_shape=(tile_h, tile_w), transform=transform)
        else: #no roads
            print('No roads')
            road_mask = np.zeros((tile_h, tile_w)) 
        
        return road_mask, intersected_buffered_lines_ser 
    
    # function that wraps from postprocessing to be used in a separate thread
    def postprocess_and_save(self, tree_and_build_mask, out_dir_root, tile_path, out_names, tile_aoi_gdf, aoi_mask, separate_masks = True):
        cfg = self.event.cfg
        road_mask = self.seg_road_tile(tile_path, aoi_mask)
        road_series = self.polyg_road_tile(tile_aoi_gdf)
        #road_mask, road_series = self.seg_and_poly_road_tile(tile_path, tile_aoi_gdf)
        tree_and_build_mask_copy = tree_and_build_mask.copy()
        overlap_masks = np.concatenate((np.expand_dims(road_mask, axis=0), tree_and_build_mask) , axis = 0)
        no_overlap_masks = segment_utils.rmv_mask_overlap(overlap_masks)
        if cfg.get('segmentation/general/clean_mask'):
            print('Cleaning the masks: holes_area_th = ', cfg.get('segmentation/general/rmv_holes_area_th'), 'small_obj_area = ', cfg.get('segmentation/general/rmv_small_obj_area_th'))
            no_overlap_masks = segment_utils.clean_masks(no_overlap_masks,
                                                         area_threshold = cfg.get('segmentation/general/rmv_holes_area_th'),
                                                         min_size = cfg.get('segmentation/general/rmv_small_obj_area_th'))
            print('Mask cleaning done')

        output.masks2Tifs(tile_path,
                        no_overlap_masks,
                        out_names = out_names,
                        separate_masks = separate_masks,
                        out_dir_root = out_dir_root)
        try:
            output.masks2parquet(tile_path, 
                                tree_and_build_mask_copy, 
                                out_dir_root=out_dir_root, 
                                out_names=out_names, road_series=road_series)
        except Exception as e:
            print(f'Error in saving parquet: {e}')

    def save_all_blank(self, out_dir_root, tile_path, out_names, separate_masks = True):
        tile_h, tile_w = samplers_utils.tile_path_2_tile_size(tile_path)
        masks = np.zeros((3, tile_h, tile_w)).astype(bool)
        output.masks2Tifs(tile_path,
                        masks,
                        out_names = out_names,
                        separate_masks = separate_masks,
                        out_dir_root = out_dir_root)

    def segment_all_tiles(self, out_dir_root, time_per_tile = []):
        mos_seg_tile = 1
        for tile_path in self.tiles_paths:
            print('')
            print(f'Starting segmenting tile {tile_path}, ({mos_seg_tile}/{self.tiles_num}), ({self.event.segmented_tiles}/{self.event.total_tiles})')
            print('')
            start_time = perf_counter()
            response = self.segment_tile(tile_path, out_dir_root=out_dir_root, separate_masks=False)
            end_time = perf_counter() 
            if response == False: #this means that buildings footprint are not available for the mosaic, go to next mosaic
                return time_per_tile, False
            execution_time = end_time - start_time 
            time_per_tile.append(execution_time)
            print(f'Finished segmenting tile {tile_path} in {execution_time:.2f} seconds')
            print(f'Average time per tile: {np.mean(time_per_tile):.2f} seconds')
            self.event.segmented_tiles += 1
            mos_seg_tile += 1
        return time_per_tile, True

class Event:
    def __init__(self,
                name,
                cfg,
                maxar_root = './data/maxar-segmentation/maxar-open-data',
                maxar_metadata_path = './metadata/from_github_maxar_metadata/datasets',
                region = 'infer'):
        #Configs
        self.cfg = cfg
        self.time_per_tile = []
        
        #esam
        self.efficient_sam = build_efficient_sam_vitt(os.path.join(self.cfg.get('models/esam/root_path'), 'weights/efficient_sam_vitt.pt')).to(self.cfg.get('models/esam/device'))        
        
        #gdino
        #self.gdino = 
        
        #Paths
        self.maxar_root = Path(maxar_root)
        self.buildings_ds_links_path = Path('./metadata/buildings_dataset_links.csv')
        self.maxar_metadata_path = Path(maxar_metadata_path)
        
        #Event
        self.name = name
        self.when = cfg.get('event/when')
        self.region_name = names.get_region_name(self.name) if region == 'infer' else region
        self.bbox = delimiters.get_event_bbox(self.name, extra_mt=1000) #TODO può essere ottimizzata sfruttando i mosaici
        self.all_mosaics_names = names.get_mosaics_names(self.name, self.maxar_root, self.when)
        
        self.wlb_gdf = gpd.read_file('./metadata/eventi_confini_complete.gpkg')
        self.filtered_wlb_gdf = self.wlb_gdf[self.wlb_gdf['event names'] == self.name]
        if self.filtered_wlb_gdf.iloc[0].geometry is None:
            print('Evento interamente su terra')
            self.cross_wlb = False
            self.filtered_wlb_gdf = None
        else:
            print('Evento su bordo')
            self.cross_wlb = True

        print(f'Creating event: {self.name}\nRegion: {self.region_name}\nMosaics: {self.all_mosaics_names}')
        #Roads
        self.road_gdf = None

        #Mosaics
        self.mosaics = {}

        #Init mosaics
        for m_name in self.all_mosaics_names:
            self.mosaics[m_name] = Mosaic(m_name, self)
        
        self.total_tiles = sum([mosaic.tiles_num for mosaic in self.mosaics.values()])
        self.segmented_tiles = 1

    def set_seg_config(self, seg_config):
        self.seg_config = seg_config
    
    #Roads methods
    def set_road_gdf(self): #set road_gdf for the event
        region_road_gdf = gen_gdf.get_region_road_gdf(self.region_name)
        self.road_gdf = filter.filter_gdf_w_bbox(region_road_gdf, self.bbox)
    
    def fast_set_road_gdf(self, roads_root = './data/microsoft-roads'):
        """
        Get a gdf containing the roads of a region.
        Input:
            region_name: Name of the region. Example: 'AfricaWest-Full'
            roads_root: Root directory of the roads datasets
        """
        start_time = time.time()
        print(f'Roads: reading roads for the whole {self.region_name} region')
        if self.region_name[-4:] != '.tsv':
            region_name = self.region_name + '.tsv'
        
        def custom_json_loads(s):
            try:
                return geometry.shape(json.loads(s)['geometry'])
            except:
                return geometry.LineString()

        chunksize = 100_000
        
        (minx, miny), (maxx, maxy) = self.bbox
        vertices = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)] #lon lat
        query_bbox_poly = geometry.Polygon(vertices)

        roads_root = Path(roads_root)
        if region_name != 'USA.tsv':
            print('Roads: not in USA. Region name:', region_name)
            for chunk in pd.read_csv(roads_root/region_name, names=['country', 'geometry'], sep='\t', chunksize=chunksize):
                hits = gbl_gdf.sindex.query(query_bbox_poly)
                gbl_gdf[hits]
            #region_road_df = pd.read_csv(roads_root/region_name, names =['country', 'geometry'], sep='\t')
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

    def set_mos_road_gdf(self, mosaic_name): #set road_gdf for the mosaic
        if self.road_gdf is None:
            self.set_road_gdf()

        self.mosaics[mosaic_name].set_road_gdf()

    def set_all_mos_road_gdf(self): #set road_gdf for all the mosaics
        for mosaic_name, mosaic in self.mosaics.items():
            if mosaic.road_gdf is None:
                self.set_mos_road_gdf(mosaic_name)
    
    #Buildings methods
    def set_build_gdf_in_mos(self, mosaic_name):
        self.mosaics[mosaic_name].set_build_gdf()

    def set_build_gdf_all_mos(self):
        for mosaic_name, mosaic in self.mosaics.items():
            if mosaic.build_gdf is None:
                self.set_build_gdf_in_mos(mosaic_name)

    def get_roi_polygon(self, wkt: bool = False):
        poly = samplers_utils.xyxy_2_Polygon(self.bbox)
        if wkt:
            return poly.wkt
        else:
            return poly
    
    def get_mosaic(self, mosaic_name):
        return self.mosaics[mosaic_name]

    #Segment methods
    def seg_all_mosaics(self, out_dir_root):
        mos_count = 1
        for __, mosaic in self.mosaics.items():
            if mosaic.is_rgb:
                print(f"Start segmenting mosaic: {mosaic.name}, ({mos_count}/{len(self.mosaics)})")
                times, response = mosaic.segment_all_tiles(out_dir_root=out_dir_root, time_per_tile=self.time_per_tile)
                self.time_per_tile.extend(times)
                mos_count += 1
                if response == False:
                    print(f'Buildings footprint not available for mosaic: {mosaic.name}. Proceeding to next mosaic...')
                    self.segmented_tiles += mosaic.tiles_num
                    continue
            else:
                print(f"First image of mosaic {mosaic.name} is not rgb, we assume the whole mosaic is not rgb. Skipping it...")
                mos_count += 1
                self.segmented_tiles += mosaic.tiles_num
                continue
        
    def seg_mos_by_keys(self, keys, out_dir_root):
        mos_count = 1
        for mos_name in keys:
            mosaic = self.mosaics[mos_name]
            if mosaic.is_rgb:
                print(f"Start segmenting mosaic: {mosaic.name}, ({mos_count}/{len(keys)})")
                times, response = mosaic.segment_all_tiles(out_dir_root=out_dir_root, time_per_tile=self.time_per_tile)
                self.time_per_tile.extend(times)
                mos_count += 1
                if response == False:
                    print(f'Buildings footprint not available for mosaic: {mosaic.name}. Proceeding to next mosaic...')
                    self.segmented_tiles += mosaic.tiles_num
                    continue
            else:
                print(f"First image of mosaic {mosaic.name} is not rgb, we assume the whole mosaic is not rgb. Skipping it...")
                mos_count += 1
                self.segmented_tiles += mosaic.tiles_num
                continue