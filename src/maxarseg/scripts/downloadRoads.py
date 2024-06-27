import pandas as pd
import os
import requests
import zipfile
import io
from pathlib import Path
import argparse

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith("_"):
            new_filename = filename.replace("_", "")
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")
            
def download_roads(meta_root, output_folder):
    meta_root = Path(meta_root)
    road_links_df = pd.read_csv( meta_root / 'roads_links.csv')
    output_folder = Path(output_folder)

    for i, row in road_links_df.iterrows(): 
        url = row['link']
        filename = url.split("/")[-1].replace("zip", "tsv")
        print(filename)
        if not os.path.exists(output_folder / filename):
            response = requests.get(url)
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            zip_file.extractall(path=output_folder)
    rename_files(output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_root', type=str, default='./metadata')
    parser.add_argument('--output_folder', type=str, default='./data/microsoft-roads')
    args = parser.parse_args()
    download_roads(args.meta_root, args.output_folder)