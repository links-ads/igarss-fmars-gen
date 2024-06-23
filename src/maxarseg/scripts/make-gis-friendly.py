"""
This snippet demonstrates how to access and convert the buildings
data from .csv.gz to geojson for use in common GIS tools. You will
need to install pandas, geopandas, and shapely.
from https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/scripts/make-gis-friendly.py
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm

def main():
    # this is the name of the geography you want to retrieve. update to meet your needs
    location = 'Morocco'

    dataset_links = pd.read_csv("/nfs/home/vaschetti/maxarSrc/metadata/buildings_dataset_links_24_05_08.csv")
    print("/nfs/home/vaschetti/maxarSrc/metadata/buildings_dataset_links_24_05_08.csv")
    greece_links = dataset_links[dataset_links.Location == location]
    for _, row in tqdm(greece_links.iterrows()):
        df = pd.read_json(row.Url, lines=True)
        df['geometry'] = df['geometry'].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        gdf.to_file(f"/nfs/home/vaschetti/maxarSrc/morocco_builds_24_05_08/{row.QuadKey}.geojson", driver="GeoJSON")
    print("Done")


if __name__ == "__main__":
    main()