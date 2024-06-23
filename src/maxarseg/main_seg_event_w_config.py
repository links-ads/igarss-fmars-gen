import argparse
import torch

from maxarseg.assemble import names
from maxarseg.assemble import holders
from maxarseg.configs import Config

torch.set_float32_matmul_precision('medium')

def main(): 
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    parser.add_argument('--config', required=True, type = str, help='Path to the custom configuration file')
    parser.add_argument('--event_ix', type = int, help='Index of the event in the list events_names')
    parser.add_argument('--out_dir_root', help='output directory root')

    args = parser.parse_args()
    
    cfg = Config(config_path = args.config)
    
    if args.event_ix is not None:
        cfg.set('event/ix', args.event_ix)
        
    if args.out_dir_root is not None:
        cfg.set('output/out_dir_root', args.out_dir_root)
        
    
    # check if there is cuda, otherwise use cpu
    if not torch.cuda.is_available():
        cfg.set('models/gd/device', 'cpu')
        cfg.set('models/df/device', 'cpu')
        cfg.set('models/esam/device', 'cpu')
    
    print(cfg._data)
    events_names = names.get_all_events()    
    event = holders.Event(events_names[cfg.get('event/ix')], cfg = cfg)
    print("Selected Event: ", event.name)
    
    #all_mosaics_names = event.all_mosaics_names
    #m0 = event.mosaics[all_mosaics_names[0]]
    #tile_path = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/southafrica-flooding22/pre/105001002B1CF200/213113323300.tif'
    event.seg_all_mosaics(out_dir_root=cfg.get('output/out_dir_root'))
    #m0.segment_all_tiles(out_dir_root=args.out_dir_root) #this segment all tiles in the mosaic
    #m0.segment_tile(tile_path, args.out_dir_root, separate_masks = False)

if __name__ == "__main__":
    main()