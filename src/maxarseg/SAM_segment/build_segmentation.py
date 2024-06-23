import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import torch


def building_gdf(country, csv_path = './metadata/buildings_dataset_links.csv', dataset_crs = None, quiet = False):
    """
    Returns a geodataframe with the buildings of the country passed as input.
    It downloads the dataset from a link in the dataset-links.csv file.
    Coordinates are converted in the crs passed as input.
    Inputs:
        country: the country of which to download the buildings. Example: 'Tanzania'
        root: the root directory of the dataset-links.csv file
        dataset_crs: the crs in which to convert the coordinates of the buildings
        quiet: if True, it doesn't print anything
    """
    dataset_links = pd.read_csv(csv_path)
    country_links = dataset_links[dataset_links.Location == country]
    #TODO: eventualmente filtrare anche sul quadkey dell evento
    if not quiet:
        print(f"Found {len(country_links)} links for {country}")

    gdfs = []
    for _, row in country_links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(shape)
        gdf_down = gpd.GeoDataFrame(df, crs=4326)
        gdfs.append(gdf_down)

    gdfs = pd.concat(gdfs)
    if dataset_crs is not None: #se inserito il crs del dataset, lo converto
        gdfs = gdfs.to_crs(dataset_crs)
    return gdfs


def rel_polyg_coord(geodf:gpd.GeoDataFrame,
                    ref_coords:tuple,
                    res):
    """
    Returns the relative coordinates of a polygon w.r.t. a reference bbox.
    Goes from absolute geo coords to relative coords in the image.

    Inputs:
        geodf: dataframe with polygons in the 'geometry' column
        ref_coords: a tuple in the format (minx, miny, maxx, maxy)
        res: resolution of the image
    Returns:
        a list of lists of tuples with the relative coordinates of the bboxes [[(p1_minx1, p1_miny1), (p1_minx2, p1_miny2), ...], [(p2_minx1, p2_miny1), (p2_minx2, p2_miny2), ...], ...]
    """
    result = []
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner

    for geom in geodf['geometry']:
        x_s, y_s = geom.exterior.coords.xy
        rel_x_s = (np.array(x_s) - ref_minx) / res
        rel_y_s = (ref_maxy - np.array(y_s)) / res
        rel_coords = list(zip(rel_x_s, rel_y_s))
        result.append(rel_coords)
    return result

def segment_buildings(predictor, building_boxes, img4Sam: np.array, use_bbox = True, use_center_points = False):
    """
    Segment the buildings in the image using the predictor passed as input.
    The image has to be encoded the image before calling this function.
    Inputs:
        predictor: the predictor to use for the segmentation
        building_boxes: a list of tuples containing the building's bounding boxes in formtat (minx, miny, maxx, maxy) = (top left corner, bottom right corner)
        img4Sam: the image previously encoded
        use_bbox: if True, the bounding boxes are used for the segmentation
        use_center_points: if True, the center points of the bounding boxes are used for the segmentation

    Returns:
        mask: a np array of shape (1, h, w). The mask is True where there is a building, False elsewhere
        bboxes: a list of tuples containing the bounding boxes of the buildings used for the segmentation
        #!used_points: a np array of shape (n, 2) where n is the number of buildings. The array contains the center points of the bounding boxes of the buildings in the image
    """

    building_boxes_t = torch.tensor(building_boxes, device=predictor.device)
    
    transformed_boxes = None
    if use_bbox:
        transformed_boxes = predictor.transform.apply_boxes_torch(building_boxes_t, img4Sam.shape[:2])
    
    transformed_points = None
    transformed_point_labels = None
    """if use_center_points: #TODO: aggiustare l'utilizzo di punti, al momento non funziona
        point_coords = torch.tensor([[(sublist[0] + sublist[2])/2, (sublist[1] + sublist[3])/2] for sublist in building_boxes_t], device=predictor.device)
        point_labels = torch.tensor([1] * point_coords.shape[0], device=predictor.device)[:, None]
        transformed_points = predictor.transform.apply_coords_torch(point_coords, img4Sam.shape[:2]).unsqueeze(1)
        transformed_point_labels = point_labels[:, None]"""

    masks, _, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=transformed_point_labels,
                boxes=transformed_boxes,
                multimask_output=False,
            )
    #mask is a tensor (n, 1, h, w) where n = number of mask = numb. of input boxes
    mask = np.any(masks.cpu().numpy(), axis = 0)

    used_boxes = None
    if use_bbox:
        used_boxes = building_boxes

    used_points = None
    """if use_center_points:
        used_points = point_coords.cpu().numpy()"""

    return mask, used_boxes, used_points #returns all the np array
