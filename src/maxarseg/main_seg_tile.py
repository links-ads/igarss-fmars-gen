import argparse

from maxarseg.assemble import names
from maxarseg.assemble import holders
from maxarseg.configs import SegmentConfig

def main(): 

    events_names = names.get_all_events()
    
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    #event
    parser.add_argument('--event_ix', default = 6, type = int, help='Index of the event in the list events_names')
    parser.add_argument('--when', default = 'pre', choices=['pre', 'post', 'None'], help='Select the pre or post event mosaics')
    
    #CONFIG
    parser.add_argument('--bs', default = 2, type = int, help = 'Batch size for the dataloader')
    parser.add_argument('--device', default = 'cuda:0', help='device to use')
    #patch
    parser.add_argument('--size', default = 600, type = int, help = 'Size of the patch')
    parser.add_argument('--stride', default = 400, type = int, help = 'Stride of the patch')
    
    #Grounding Dino - Trees
    parser.add_argument('--GD_root', default = "./models/GDINO", help = 'Root of the grounding dino model')
    parser.add_argument('--GD_config_file', default = "GroundingDINO_SwinT_OGC.py", help = 'Config file of the grounding dino model')
    parser.add_argument('--GD_weights', default = "groundingdino_swint_ogc.pth", help = 'Weights of the grounding dino model')
    
    parser.add_argument('--text_prompt', default = 'green tree', help = 'Prompt for the grounding dino model')
    parser.add_argument('--box_threshold', default = 0.15, type = float, help = 'Threshold for the grounding dino model')
    parser.add_argument('--text_threshold', default = 0.30, type = float, help = 'Threshold for the grounding dino model')
    parser.add_argument('--max_area_GD_boxes_mt2', default = 6000, type = int, help = 'Max area of the boxes for the grounding dino model')
    
    parser.add_argument('--min_ratio_GD_boxes_edges', default = 0, type = float, help = 'Min ratio between edges of the tree boxes')
    parser.add_argument('--perc_reduce_tree_boxes', default = 0, type = float, help = 'Percentage of reduction of the tree boxes')
    
    #Buildings
    parser.add_argument('--ext_mt_build_box', default = 5, type = int, help = 'Width of the building')
    
    #Roads
    parser.add_argument('--road_width_mt', default = 5, type = int, help = 'Width of the road')
    
    
    #Efficient SAM
    parser.add_argument('--ESAM_root', default = './models/EfficientSAM', help = 'Root of the efficient sam model')
    parser.add_argument('--ESAM_num_parall_queries', default = 5, type = int, help = 'Set the number of paraller queries to be processed')
    
    
    parser.add_argument('--out_dir_root', default = "./output/tiff", help='output directory root')

    args = parser.parse_args()
        
    print("Selected Event: ", events_names[args.event_ix])
    
    config = SegmentConfig(batch_size = args.bs,
                           size = args.size,
                           stride = args.stride,
                           
                           device = args.device,
                           GD_root = args.GD_root,
                           GD_config_file = args.GD_config_file,
                           GD_weights = args.GD_weights,
                           
                           TEXT_PROMPT = args.text_prompt,
                           BOX_THRESHOLD = args.box_threshold,
                           TEXT_THRESHOLD = args.text_threshold,
                           
                           max_area_GD_boxes_mt2 = args.max_area_GD_boxes_mt2,
                           min_ratio_GD_boxes_edges = args.min_ratio_GD_boxes_edges,
                           perc_reduce_tree_boxes = args.perc_reduce_tree_boxes,
                           
                           road_width_mt=args.road_width_mt,
                           ext_mt_build_box=args.ext_mt_build_box,
                           
                           ESAM_root = args.ESAM_root,
                           ESAM_num_parall_queries = args.ESAM_num_parall_queries,
                           )
    
    event = holders.Event(events_names[args.event_ix], seg_config = config, when=args.when)
    all_mosaics_names = event.all_mosaics_names
    
    
    #event.seg_all_mosaics() #this segment all the mosiacs in the event
    
    m0 = event.mosaics[all_mosaics_names[0]]
    #m0.segment_all_tiles() #this segment all tiles in the mosaic
    
    m0_tile_17_path = m0.tiles_paths[17]
    m0.segment_tile(m0_tile_17_path, args.out_dir_root)


if __name__ == "__main__":
    main()