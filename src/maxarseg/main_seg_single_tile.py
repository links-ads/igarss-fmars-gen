import argparse
import torch

from maxarseg.assemble import names
from maxarseg.assemble import holders
from maxarseg.configs import SegmentConfig, DetectConfig

torch.set_float32_matmul_precision('medium')

def main(): 

    events_names = names.get_all_events()
    
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    #event
    parser.add_argument('--event_ix', default = 2, type = int, help='Index of the event in the list events_names')
    parser.add_argument('--when', default = 'pre', choices=['pre', 'post', 'None'], help='Select the pre or post event mosaics')
    
    #Detect config
    parser.add_argument('--GD_bs', default = 1, type = int, help = 'Batch size for Grounding Dino')
    parser.add_argument('--DF_bs', default = 32, type = int, help = 'Batch size for DeepForest')

    parser.add_argument('--device_det', default = 'cuda:0', help='device to use for detection')
    
    parser.add_argument('--size_det', default = 600, type = int, help = 'Size of the patch for detection')
    parser.add_argument('--stride_det', default = 400, type = int, help = 'Stride of the patch for detection')
    
    parser.add_argument('--GD_root', default = "./models/GDINO", help = 'Root of the grounding dino model')
    parser.add_argument('--GD_config_file', default = "configs/GroundingDINO_SwinT_OGC.py", help = 'Config file of the grounding dino model')
    parser.add_argument('--GD_weights', default = "weights/groundingdino_swint_ogc.pth", help = 'Weights of the grounding dino model')
    
    parser.add_argument('--text_prompt', default = 'bush', help = 'Prompt for the grounding dino model')
    parser.add_argument('--box_threshold', default = 0.12, type = float, help = 'Threshold for the grounding dino model')
    parser.add_argument('--text_threshold', default = 0.30, type = float, help = 'Threshold for the grounding dino model')
    
    parser.add_argument('--max_area_GD_boxes_mt2', default = 6000, type = int, help = 'Max area of the boxes for the grounding dino model')
    parser.add_argument('--min_ratio_GD_boxes_edges', default = 0.5, type = float, help = 'Min ratio between edges of the tree boxes')
    parser.add_argument('--perc_reduce_tree_boxes', default = 0, type = float, help = 'Percentage of reduction of the tree boxes')
    
    #Segment config
    parser.add_argument('--bs_seg', default = 1, type = int, help = 'Batch size for the segmentation')
    parser.add_argument('--device_seg', default = 'cuda:0', help='device to use')
    
    parser.add_argument('--size_seg', default = 1024, type = int, help = 'Size of the patch')
    parser.add_argument('--stride_seg', default = 1024 - 256, type = int, help = 'Stride of the patch')
    
    parser.add_argument('--ext_mt_build_box', default = 0, type = int, help = 'Extra meter to enlarge building boxes')
    
    parser.add_argument('--road_width_mt', default = 5, type = int, help = 'Width of the road')    
    
    #Efficient SAM
    parser.add_argument('--ESAM_root', default = './models/EfficientSAM', help = 'Root of the efficient sam model')
    parser.add_argument('--ESAM_num_parall_queries', default = 5, type = int, help = 'Set the number of paraller queries to be processed')
    parser.add_argument('--out_dir_root', default = "./output/tiff/prova_write_canvas", help='output directory root')

    args = parser.parse_args()
        
    print("Selected Event: ", events_names[args.event_ix])
    
    # check if there is cuda, otherwise use cpu
    if not torch.cuda.is_available():
        args.device_det = 'cpu'
        args.device_seg = 'cpu'
    
    det_config = DetectConfig(
                            GD_batch_size = args.GD_bs,
                            DF_batch_size = args.DF_bs,
                            size = args.size_det,
                            stride = args.stride_det,
                            device = args.device_det,
                            GD_root = args.GD_root,
                            GD_config_file = args.GD_config_file,
                            GD_weights = args.GD_weights,
                            TEXT_PROMPT = args.text_prompt,
                            max_area_GD_boxes_mt2 = args.max_area_GD_boxes_mt2,
                            min_ratio_GD_boxes_edges = args.min_ratio_GD_boxes_edges,
                            perc_reduce_tree_boxes = args.perc_reduce_tree_boxes,
                            )
    
    seg_config = SegmentConfig(batch_size = args.bs_seg,
                                size = args.size_seg,
                                stride = args.stride_seg,
                                device = args.device_seg,
                                road_width_mt=args.road_width_mt,
                                ext_mt_build_box=args.ext_mt_build_box,
                                ESAM_root = args.ESAM_root,
                                ESAM_num_parall_queries = args.ESAM_num_parall_queries,
                                use_separate_detect_config=True,
                                clean_masks_bool= True
                                )
    
    event = holders.Event(events_names[args.event_ix],
                        seg_config = seg_config,
                        det_config = det_config,
                        when=args.when)
    
    all_mosaics_names = event.all_mosaics_names
        
    m0 = event.mosaics[all_mosaics_names[1]]
    
    land_and_water_tile_path = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/Gambia-flooding-8-11-2022/pre/105001002BD68F00/033133031231.tif'
    only_water_tile_path = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/Gambia-flooding-8-11-2022/pre/105001002BD68F00/033133031303.tif'
    tile_path = m0.tiles_paths[2]
    m0.segment_tile(tile_path, args.out_dir_root, separate_masks = False, overwrite = True)

if __name__ == "__main__":
    main()