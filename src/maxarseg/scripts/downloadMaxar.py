import leafmap
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import os
import argparse
from pathlib import Path

if Path.cwd().name != 'src':
    os.chdir('./src')

# Read the csv file
events_df = pd.read_csv('../metadata/dateEventi.csv', sep=';')
# Create a dictionary with the event name as key and the date as value
event2date = events_df.set_index('Aligned name')['date'].to_dict()


def get_pre_post_gdf_local(collection_id, event2date = event2date, local_gdf = True):
    
    #Retrieve the event date
    try:
        event_date = event2date[collection_id]
    except:
        print("ERROR: Event date not found!!!")
        return None, None

    #Create the geodataframe
    geojson_path = '../metadata/from_github_maxar_metadata/datasets'
    if local_gdf:
        gdf = gpd.read_file(os.path.join(geojson_path, collection_id + '.geojson'))
    else:
        gdf = gpd.GeoDataFrame()
        for child_id in tqdm(leafmap.maxar_child_collections(collection_id)):
            current_gdf = leafmap.maxar_items(
                collection_id = collection_id,
                child_id = child_id,
                return_gdf=True,
                assets=['visual'],
            )
        gdf = pd.concat([gdf, current_gdf])
        
    
    #Split the geodataframe
    pre_gdf = gdf[gdf['datetime'] < event_date]
    post_gdf = gdf[gdf['datetime'] >= event_date]

    print('Collection_id:',collection_id,'\nEvent date:', event_date)

    if pre_gdf.shape[0] + post_gdf.shape[0] == gdf.shape[0]:
        print("OK: All items are accounted for\n")
    else:
        print("ERROR: Some items are missing!!!\n")

    print("pre_gdf", pre_gdf.shape)
    print("post_gdf", post_gdf.shape)

    return pre_gdf, post_gdf

def download_event(collection_id, out_dir_root = "../data/maxar-open-data/"):
    
    pre_gdf, post_gdf = get_pre_post_gdf_local(collection_id)
    if pre_gdf is None or post_gdf is None:
        return

    leafmap.maxar_download(pre_gdf['visual'].to_list(), out_dir = os.path.join(out_dir_root, collection_id, 'pre', ""))
    leafmap.maxar_download(post_gdf['visual'].to_list(), out_dir = os.path.join(out_dir_root, collection_id, 'post', ""))
    

def main():
    pareser = argparse.ArgumentParser(description='Download Maxar images')
    pareser.add_argument('--c_id', help='single or list of collection id you want to download')
    pareser.add_argument('--out_dir', default = "../data/maxar-open-data/", help='output directory')

    args = pareser.parse_args()

    if args.c_id is None:
        collection_ids = leafmap.maxar_collections()
    else:
        collection_ids = [args.c_id]

    for collection_id in collection_ids:
        download_event(collection_id, args.out_dir)


if __name__ == '__main__':
    main()