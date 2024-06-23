import argparse
import torch

from maxarseg.assemble import names
from maxarseg.assemble import holders
from maxarseg.configs import Config

torch.set_float32_matmul_precision('medium')

def main(): 
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    parser.add_argument('--config', required= True, type = str, help='Path to the custom configuration file')
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
    
    all_mosaics_names = event.all_mosaics_names
    m0 = event.mosaics[all_mosaics_names[0]] #bay of bengal or Gambia
    print("Selected Mosaic: ", m0.name)
    
    land_and_water_tile_path = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/Gambia-flooding-8-11-2022/pre/105001002BD68F00/033133031231.tif'
    only_water_tile_path = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/Gambia-flooding-8-11-2022/pre/105001002BD68F00/033133031303.tif'
    
    #tile_path = m0.tiles_paths[22] #bay of bengal
    
    print("Selected Tile: ", land_and_water_tile_path)
    m0.segment_tile(land_and_water_tile_path, args.out_dir_root, separate_masks = False, overwrite = True)

if __name__ == "__main__":
    main()