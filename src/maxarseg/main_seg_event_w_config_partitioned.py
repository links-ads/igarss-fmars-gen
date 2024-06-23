import argparse
from re import I
import torch
import math

from maxarseg.assemble import names
from maxarseg.assemble import holders
from maxarseg.configs import Config

torch.set_float32_matmul_precision('medium')

def balance_groups(nums, k):
    # Sort numbers in descending order
    nums.sort(reverse=True)
    
    # Initialize k groups
    groups = [[] for _ in range(k)]
    sums = [0] * k
    
    # Distribute numbers into groups
    for num in nums:
        # Find the group with the smallest sum and add the number to it
        idx = sums.index(min(sums))
        groups[idx].append(num)
        sums[idx] += num
    
    return groups

def main(): 
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    parser.add_argument('--config', required=True, type = str, help='Path to the custom configuration file')
    parser.add_argument('--event_ix', type = float, help='Index of the event in the list events_names')
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
    
    if '.' not in str(cfg.get('event/ix')):
        raise ValueError("Event ix should contain a sub partition. E.g. event/ix = 4.1")
    
    mos_partition = int(str(cfg.get('event/ix')).split('.')[-1])
    event_ix = int(str(cfg.get('event/ix')).split('.')[0])
    print(cfg._data)
    events_names = names.get_all_events()    
    event = holders.Event(events_names[event_ix], cfg = cfg)
    print("Selected Event: ", event.name)
    
    all_mosaics_names = event.all_mosaics_names
    all_mosaics_names.sort()
    
    #There always be 10 partition x.0 to x.9
    mos_ix_s = math.floor(len(all_mosaics_names)/10) * mos_partition
    if mos_partition == 9:
        mos_ix_e = len(all_mosaics_names)
    else:
        mos_ix_e = math.floor(len(all_mosaics_names)/10) * (mos_partition + 1)
        
    mos_names = all_mosaics_names[mos_ix_s:mos_ix_e]
    
    part_num_tiles = sum([event.mosaics[k].tiles_num for k in mos_names])
    print()
    print(f'This partition will segment {len(mos_names)}/{len(event.mosaics)} mosaics. {part_num_tiles}/{event.total_tiles} imgs in total')
    print()
    event.seg_mos_by_keys(keys = mos_names, out_dir_root=cfg.get('output/out_dir_root'))

if __name__ == "__main__":
    main()

