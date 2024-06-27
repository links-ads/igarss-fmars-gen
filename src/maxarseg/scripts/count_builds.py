#%%
from maxarseg.assemble import holders
import os
from pathlib import Path
from maxarseg.assemble import delimiters, names
import geopandas as gpd
import numpy as np
from maxarseg.samplers import samplers_utils
import time
import pandas as pd
import sys
import rasterio

def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
def list_tif_files(path):
    return [f for f in os.listdir(path) if f.endswith('.tif') and os.path.isfile(os.path.join(path, f))]

def list_parquet_files(path):
    return [f for f in os.listdir(path) if f.endswith('.parquet') and os.path.isfile(os.path.join(path, f))]

def filter_gdf_vs_aois_gdf(proj_gdf, aois_gdf):
    num_hits = np.array([0]*len(proj_gdf))
    for geom in aois_gdf.geometry:
        hits = proj_gdf.intersects(geom)
        num_hits = num_hits + hits.values
    return proj_gdf[num_hits >= 1]

class Event_light:
    def __init__(self,
                name,
                maxar_root = './data/maxar-open-data',
                maxar_metadata_path = './metadata/from_github_maxar_metadata/datasets',
                region = 'infer'):

        #Paths
        self.maxar_root = Path(maxar_root)
        self.buildings_ds_links_path = Path('./metadata/buildings_dataset_links.csv')
        self.maxar_metadata_path = Path(maxar_metadata_path)
        
        #Event
        self.name = name
        self.when = 'pre'
        self.region_name = names.get_region_name(self.name) if region == 'infer' else region
        self.bbox = delimiters.get_event_bbox(self.name, extra_mt=1000)
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
            self.mosaics[m_name] = holders.Mosaic(m_name, self)
        
        self.total_tiles = sum([mosaic.tiles_num for mosaic in self.mosaics.values()])
        
    def __str__(self) -> str:
        res = f'\n_______________________________________________________\nEvent: {self.name}\nMosaics: {self.all_mosaics_names}\nTotal tiles: {self.total_tiles}\n_______________________________________________________\n'
        return res

#%%
def main():
    print('Starting...', flush=True)
    lbl_root_folder = '/nfs/projects/overwatch/maxar-segmentation/outputs/04_05/train' #cartella delle label
    ev_names = list_directories(lbl_root_folder)
    ev_names = sorted(ev_names)
    cols = {'event': [],
            'mosaic': [],
            'tile': [],
            'num_ms_build_aoi': [],
            'num_ms_build_aoi_no_water_for': [],
            'num_ms_build_aoi_no_water_sjoin': [],
            'parquet_build': [],
            'parquet_tree': [],
            'bg_pxl': [],
            'road_pxl': [],
            'tree_pxl': [],
            'build_pxl': [],
            'entropy': []
            }
    high_res_entropies = []
    len_ev = len(ev_names)
    current_ev = 0
    for ev_name in ev_names:
        if ev_name == 'Morocco-Earthquake-Sept-2023' or ev_name == 'Hurricane-Ian-9-26-2022':
            continue
        current_ev += 1
        mos_names = list_directories(os.path.join(lbl_root_folder, ev_name, 'pre'))
        mos_names = sorted(mos_names)
        if len(mos_names) > 0:
            event = Event_light(ev_name) #this event will contain all the events in the imgs
            print(event, flush=True)
        
        current_mos = 0
        for mos_name in mos_names: #only mos in lbl
            current_mos += 1
            print(f'\n{ev_name}/{mos_name}', flush=True)
            mos = event.mosaics[mos_name]
            try:
                mos.set_build_gdf()
                print(f'len build gdf {len(mos.build_gdf):,}', flush=True)
            except:
                print(f'No buildings in {ev_name}/{mos_name}', file=sys.stderr)
                continue
            tif_names = list_tif_files(os.path.join(lbl_root_folder, ev_name, 'pre', mos_name))
            tif_names = sorted(tif_names)
            parquets_names = list_parquet_files(os.path.join(lbl_root_folder, ev_name, 'pre', mos_name))
            if len(tif_names) != len(parquets_names):
                print(f'{ev_name}/{mos_name}. Not corresponcence between tifs and parquet, lists are not of equal length.', file=sys.stderr)

            print('proc_tifs and parquet:', len(tif_names), flush=True)
            print()
            current_tif = 0
            for tile_name in tif_names:
                current_tif += 1
                print(f'{ev_name}/pre/{mos_name}/{tile_name}, tif:({current_tif}/{len(tif_names)}), mos:({current_mos}/{len(mos_names)}), ev:({current_ev}/{len_ev})', flush=True)
                cols['event'].append(ev_name)
                cols['mosaic'].append(mos_name)
                cols['tile'].append(tile_name)
                
                tile_path = os.path.join(lbl_root_folder, ev_name, 'pre', mos_name, tile_name)
                parquet_path = tile_path[:-4] + '.parquet'
                #tile_aoi = gpd.GeoDataFrame({'geometry': [samplers_utils.path_2_tile_aoi(tile_path)]})
                
                num_aoi_build = len(mos.proj_build_gdf.iloc[mos.sindex_proj_build_gdf.query(samplers_utils.path_2_tile_aoi(tile_path))])
                cols['num_ms_build_aoi'].append(num_aoi_build)
                #print('tile_aoi builds', num_aoi_build)
                #print()
                
                tile_aoi_no_water = samplers_utils.path_2_tile_aoi_no_water(tile_path, event.filtered_wlb_gdf)
                num_ms_build_aoi_no_water_for = len(filter_gdf_vs_aois_gdf(mos.proj_build_gdf, tile_aoi_no_water))
                cols['num_ms_build_aoi_no_water_for'].append(num_ms_build_aoi_no_water_for)
                #print('tile_aoi builds no water filter with for', num_ms_build_aoi_no_water_for)
                #print()
                
                num_ms_build_aoi_no_water_sjoin = len(gpd.sjoin(mos.proj_build_gdf, tile_aoi_no_water, how='inner', op='intersects'))
                cols['num_ms_build_aoi_no_water_sjoin'].append(num_ms_build_aoi_no_water_sjoin)
                #print('tile_aoi builds no water filter with sjoin', num_ms_build_aoi_no_water_sjoin)
                #print()
                if os.path.exists(parquet_path):
                    parquet_build = sum(pd.read_parquet(parquet_path, engine='pyarrow').class_id == 2)
                    parquet_trees = sum(pd.read_parquet(parquet_path, engine='pyarrow').class_id == 1)
                else:
                    print(f'{parquet_path} do not exists', file=sys.stderr)
                    parquet_build = -1
                    parquet_trees = -1
                cols['parquet_build'].append(parquet_build)
                cols['parquet_tree'].append(parquet_trees)

                
                with rasterio.open(tile_path) as src: #here read the lbl
                    lbl = src.read()
                tot_pxl = lbl.shape[-1]**2 #assume every img is squared
                cols['bg_pxl'].append(np.sum(lbl == 255)/tot_pxl)
                cols['road_pxl'].append(np.sum(lbl == 0)/tot_pxl)
                cols['tree_pxl'].append(np.sum(lbl == 1)/tot_pxl)
                cols['build_pxl'].append(np.sum(lbl == 2)/tot_pxl)
                cols['entropy'].append(samplers_utils.entropy_from_lbl(lbl))
                
                high_res_entropies.append(samplers_utils.compute_entropy_matrix(lbl))

        #if current_ev == 2:
        #    break
    high_res_entropies_np = np.stack(high_res_entropies, axis=0)
    np.save('high_res_entropies.npy', high_res_entropies_np)
    print('Saved high_res_entropies.npy', flush=True)
    res_df = pd.DataFrame(cols)
    res_df.to_csv('lbl_stats.csv', index = True)
    print('Saved lbl_stats.csv', flush=True)

if __name__ == "__main__":
    main()
