from maxarseg import build
from tqdm import tqdm
import pandas as pd

def main():
    df = pd.read_csv('./output/stats_roadnBuild.csv')
    print(df)
    tot_buildings = 0
    tot_roads = 0 
    config = build.SegmentConfig(batch_size = 4, device='cpu')
    for event_name in tqdm(build.get_all_events()):
        if event_name in df['event_name'].values:
            continue
        try:
            event_build_num = 0
            event_road_num = 0
            print('\n', event_name)
            evento = build.Event(event_name, seg_config = config, when='pre')
            evento.set_all_mos_road_gdf()
            evento.set_build_gdf_all_mos()
            for _, mos in evento.mosaics.items():
                event_build_num += mos.build_num
                event_road_num += mos.road_num
            
            new_row = pd.DataFrame({'event_name': [event_name], 'num_road': [event_road_num], 'num_build': [event_build_num]})
            df = pd.concat([df, new_row], ignore_index=True)
            tot_buildings += event_build_num
            tot_roads += event_road_num
            print(f'Event: {event_name}, Total buildings: {event_build_num}, Total roads: {event_road_num}')
        except Exception as e:
            print(f'Error in {event_name}')
            df.to_csv('./output/stats_roadnBuild.csv', index=False)
            print(f"Caught an exception: {e}")
            return

    print('Total buildings: ', tot_buildings)
    print('Total roads: ', tot_roads)
    df.to_csv('./output/stats_roadnBuild.csv', index=False)
    return

if __name__ == "__main__":
    main()
    