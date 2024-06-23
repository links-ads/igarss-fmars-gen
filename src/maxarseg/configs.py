
from pathlib import Path
import os
import sys
import yaml


from groundingdino.util.inference import load_model as GD_load_model
from maxarseg.efficient_sam.build_efficient_sam import build_efficient_sam_vitt


class SegmentConfig:
    """
    Config class for the segmentation pipeline.
    It contains detection and segmentation parameters as well as the models themselves. 
    """
    def __init__(self,
                batch_size,
                size = 600,
                stride = 400,
                
                device = 'cuda',
                GD_root = "./models/GDINO",
                GD_config_file = "GroundingDINO_SwinT_OGC.py",
                GD_weights = "groundingdino_swint_ogc.pth",
                
                TEXT_PROMPT = 'bush', #'green tree'
                BOX_THRESHOLD = 0.15,
                TEXT_THRESHOLD = 0.30,
                
                max_area_GD_boxes_mt2 = 6000,
                min_ratio_GD_boxes_edges = 0,
                perc_reduce_tree_boxes = 0,
                
                road_width_mt = 5,
                ext_mt_build_box = 0,
                
                ESAM_root = './models/EfficientSAM',
                ESAM_num_parall_queries = 5,
                smooth_patch_overlap = False, #if this is false, stride could be equal to size
                use_separate_detect_config = False,
                
                clean_masks_bool = False,
                rmv_holes_area_th = 80,
                rmv_small_obj_area_th = 80):
        
        #General
        self.batch_size = batch_size
        self.size = size
        self.stride = stride # Overlap between each patch = (size - stride)
        self.device = device
        self.smooth_patch_overlap = smooth_patch_overlap
        
        if not use_separate_detect_config: #if you are not using a separate detect_config then define here all the detection configuration
            #Grounding Dino (Trees)
            self.GD_root = Path(GD_root)
            self.CONFIG_PATH = self.GD_root / GD_config_file
            self.WEIGHTS_PATH = self.GD_root / GD_weights

            self.GD_model = GD_load_model(self.CONFIG_PATH, self.WEIGHTS_PATH).to(self.device)
            self.TEXT_PROMPT = TEXT_PROMPT
            self.BOX_THRESHOLD = BOX_THRESHOLD
            self.TEXT_THRESHOLD = TEXT_THRESHOLD
            self.max_area_GD_boxes_mt2 = max_area_GD_boxes_mt2
            self.min_ratio_GD_boxes_edges = min_ratio_GD_boxes_edges
            self.perc_reduce_tree_boxes = perc_reduce_tree_boxes

        #Efficient SAM
        self.efficient_sam = build_efficient_sam_vitt(os.path.join(ESAM_root, 'weights/efficient_sam_vitt.pt')).to(self.device)
        self.ESAM_num_parall_queries = ESAM_num_parall_queries
        
        #Roads
        self.road_width_mt = road_width_mt
        
        #Buildings
        self.ext_mt_build_box = ext_mt_build_box
        
        #Post proc
        self.clean_masks_bool = clean_masks_bool
        self.rmv_holes_area_th = rmv_holes_area_th
        self.rmv_small_obj_area_th = rmv_small_obj_area_th
        
        if not use_separate_detect_config:
            print('\n- GD model device:', next(self.GD_model.parameters()).device)
        print('- Efficient SAM device:', next(self.efficient_sam.parameters()).device)
    
    def __str__(self) -> str:
        return f'{self.TEXT_PROMPT = }\n{self.BOX_THRESHOLD = }\n{self.TEXT_THRESHOLD = }\n{self.max_area_GD_boxes_mt2 = }\n{self.min_ratio_GD_boxes_edges = }\n{self.perc_reduce_tree_boxes = }\n{self.road_width_mt = }\n{self.ext_mt_build_box = }'
class DetectConfig:

    def __init__(self,
                size = 600,
                stride = 400,                
                device = 'cuda',
                
                GD_batch_size = 1,
                GD_root = "./models/GDINO",
                GD_config_file = "GroundingDINO_SwinT_OGC.py",
                GD_weights = "groundingdino_swint_ogc.pth",
                
                TEXT_PROMPT = 'bush', #'green tree'
                BOX_THRESHOLD = 0.15,
                TEXT_THRESHOLD = 0.30,
                
                DF_patch_size = 800,
                DF_patch_overlap = 0.25,
                DF_box_threshold = 0.1,
                DF_batch_size = 16, 
                
                max_area_GD_boxes_mt2 = 6000,
                min_ratio_GD_boxes_edges = 0.0,
                perc_reduce_tree_boxes = 0.0,
                nms_threshold = 0.5):
        
        #General
        self.size = size
        self.stride = stride # Overlap between each patch = (size - stride)
        self.device = device
        
        #Grounding Dino (Trees)
        self.GD_batch_size = GD_batch_size
        self.GD_root = Path(GD_root)
        self.CONFIG_PATH = self.GD_root / GD_config_file
        self.WEIGHTS_PATH = self.GD_root / GD_weights

        #self.GD_model = GD_load_model(self.CONFIG_PATH, self.WEIGHTS_PATH).to(self.device)
        self.TEXT_PROMPT = TEXT_PROMPT
        self.BOX_THRESHOLD = BOX_THRESHOLD
        self.TEXT_THRESHOLD = TEXT_THRESHOLD
        
        #DeepForest
        self.DF_patch_size = DF_patch_size
        self.DF_patch_overlap = DF_patch_overlap
        self.DF_box_threshold = DF_box_threshold
        self.DF_device = [int(device.split(':')[-1])] if len(device.split(':')) > 1 else 'auto' #Remove the port number from the device (e.g. 'cuda:0' -> 'cuda')
        self.DF_batch_size = DF_batch_size
        
        #Filtering
        self.max_area_GD_boxes_mt2 = max_area_GD_boxes_mt2
        self.min_ratio_GD_boxes_edges = min_ratio_GD_boxes_edges
        self.perc_reduce_tree_boxes = perc_reduce_tree_boxes
        self.nms_threshold = nms_threshold
        
    def __str__(self) -> str:
        return f'{self.TEXT_PROMPT = }\n{self.BOX_THRESHOLD = }\n{self.TEXT_THRESHOLD = }\n{self.max_area_GD_boxes_mt2 = }\n{self.min_ratio_GD_boxes_edges = }\n{self.perc_reduce_tree_boxes = }'

def merge_dictionaries_recursively(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
    dict1 (dict): first dictionary to be updated
    dict2 (dict): second dictionary which entries should be preferred
    '''
    if dict2 is None: return

    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_dictionaries_recursively(dict1[k], v)
        else:
            dict1[k] = v

class Config(object):  
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """
    def __init__(self, config_path, default_path='./configs/default_cfg.yaml'):
        with open(config_path) as cf_file:
            cfg = yaml.safe_load( cf_file.read() )
            
        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = yaml.safe_load( def_cf_file.read() )
                
            merge_dictionaries_recursively(default_cfg, cfg)
            
        self._data = default_cfg

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        for path_item in path_items:
            sub_dict = sub_dict.get(path_item)

        value = sub_dict.get(data_item, default)
        if value is None:
            raise ValueError(f"Path '{path}' not found in config file")
        return value
        
    def set(self, path, value):
        if path is None:
            raise ValueError("Path cannot be None")

        path_items = path.split("/")
        sub_dict = self._data

        # Traverse the dictionary except for the last key
        for path_item in path_items[:-1]:
            if path_item not in sub_dict:
                sub_dict[path_item] = {}  # Create a new dictionary if the key does not exist
            sub_dict = sub_dict[path_item]

        # Set the value to the last key in the path
        sub_dict[path_items[-1]] = value
    
    def __str__(self) -> str:
        return str(self._data)