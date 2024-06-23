import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, LineString, MultiPoint, Point
from typing import List, Union
from rasterio.features import rasterize
from skimage import morphology



def rel_road_lines(geodf: gpd.GeoDataFrame,
                    query_bbox_poly: Polygon,
                    res):
    """
    Given a Geodataframe containing Linestrings with geo coords, 
    returns the relative coordinates of those lines w.r.t. a reference bbox

    Inputs:
        geodf: GeoDataFrame containing the Linestring
        query_bbox_poly: Polygon of the reference bbox
        res: resolution of the image
    Returns:
        result: list of LineString with the relative coordinates
    """
    ref_coords = query_bbox_poly.bounds
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner

    result = []
    for line in geodf.geometry:
        x_s, y_s = line.coords.xy

        rel_x_s = (np.array(x_s) - ref_minx) / res
        rel_y_s = (ref_maxy - np.array(y_s)) / res
        rel_coords = list(zip(rel_x_s, rel_y_s))
        line = LineString(rel_coords)
        result.append(line)
    return result

def line2points(lines: Union[LineString, List[LineString]], points_dist) -> List[Point]:
    """
    Given a single or a list of shapely.LineString,
    returns a list of shapely points along all the lines, spaced by points_dist
    """
    if not isinstance(lines, list):
        lines = [lines]
    points = []
    for line in lines:
        points.extend([line.interpolate(dist) for dist in np.arange(0, line.length, points_dist)])
    return points

def get_offset_lines(lines: Union[LineString, List[LineString]], distance=35):
    """
    Create two offset lines from a single or list of shapely.LineString at distance = 'distance'
    """
    if not isinstance(lines, list):
        lines = [lines]
    
    offset_lines = []
    for line in lines:
        for side in [-1, +1]:
            offset_lines.append(line.offset_curve(side*distance ))
    return offset_lines

def clear_roads(lines: Union[LineString, List[LineString]], bg_points, distance) -> List[Point]:
    """
    Given a list of shapely.LineString and a list of shapely.Point,
    remove bg points that may be on the road
    """
    candidate_bg_pts = bg_points
    final_bg_pts = set(bg_points)

    if not isinstance(lines, list):
        lines = [lines]

    for line in lines:
        line_space = line.buffer(distance)
        for point in candidate_bg_pts:
            if line_space.contains(point):
                final_bg_pts.discard(point)
        
    return list(final_bg_pts)

def rmv_rnd_fraction(points, fraction_to_keep):
    """
    Removes a random fraction of the points
    """
    np.random.shuffle(points)
    points = points[:int(len(points)*fraction_to_keep)]
    return points

def rmv_pts_out_img(points: np.array, sample_size)-> np.array:
    """
    Given a np.array of points (n, 2),
    removes points outside the image
    """
    if len(points) != 0:
        points = points[np.logical_and(np.logical_and(points[:, 0] >= 0, points[:, 0] < sample_size), np.logical_and(points[:, 1] >= 0, points[:, 1] < sample_size))]
    return points

def segment_roads(predictor,
                  road_lines: Union[LineString, List[LineString]],
                  sample_size,
                  img4Sam = None,
                  road_point_dist = 50,
                  bg_point_dist = 80,
                  offset_distance = 50,
                  do_clean_mask = True):
    """
    Segment the roads in the image using the predictor passed as input.
    If passed as input the image is encoded on the fly, otherwise it has to be encoded before calling this function.

    Inputs:
        predictor: the predictor to use for the segmentation
        road_lines: a list of shapely.LineString containing the roads
        sample_size: the size of the image
        img4Sam: the image to encode if not already encoded
        road_point_dist: the distance between two points on the road
        bg_point_dist: the distance between two points in the road's offset lines
        offset_distance: the offset distance
        do_clean_mask: if True, the mask is cleaned by removing parts outside the offset lines

    Returns:
        final_mask: a np array of shape (1, h, w). The mask is True where there is a road, False elsewhere
        final_pt_coords4Sam: a np array of shape (n, 2) where n is the number of points. The array contains the coordinates of the points used for the segmentation
        final_labels4Sam: a np array of shape (n,) where n is the number of points. The array contains the labels of the points used for the segmentation
    """
    
    #Decide if encoding here or outside the function
    if img4Sam is not None:
        predictor.set_image(img4Sam)
    
    #initialize an empty mask
    final_mask = np.full((sample_size, sample_size), False)
    
    final_pt_coords4Sam = []
    final_labels4Sam = []

    if not isinstance(road_lines, list):
        road_lines = [road_lines]

    for road in road_lines:
        road_pts = line2points(road, road_point_dist) #turn the road into a list of shapely points
        np_roads_pts = np.array([list(pt.coords)[0] for pt in road_pts]) #turn the shapely points into a numpy array
        np_roads_pts = rmv_pts_out_img(np_roads_pts, sample_size) #remove road points outside the image
        np_road_labels = np.array([1]*np_roads_pts.shape[0]) #create the labels for the road points
        
        bg_lines = get_offset_lines(road, offset_distance) #create two offset lines from the road
        bg_pts = line2points(bg_lines, bg_point_dist) #turn the offset lines into a list of shapely points
        bg_pts = clear_roads(road_lines, bg_pts, offset_distance - 4) #remove bg points that may be on other roads
        np_bg_pts = np.array([list(pt.coords)[0] for pt in bg_pts]) #turn the shapely points into a numpy array
        np_bg_pts = rmv_pts_out_img(np_bg_pts, sample_size) #remove road points outside the image
        np_bg_labels = np.array([0]*np_bg_pts.shape[0]) #create the labels for the bg points

        if len(np_bg_labels) == 0 or len(np_road_labels) < 2: #if there are no bg_points or 0 or 1 road points skip the road
            continue

        pt_coords4Sam = np.concatenate((np_roads_pts, np_bg_pts)) #tmp list
        labels4Sam = np.concatenate((np_road_labels, np_bg_labels))

        final_pt_coords4Sam.extend(pt_coords4Sam.tolist()) #global list
        final_labels4Sam.extend(labels4Sam.tolist()) #global list

        mask, _, _ = predictor.predict(
                point_coords=pt_coords4Sam,
                point_labels=labels4Sam,
                multimask_output=False,
            )
        final_mask = np.logical_or(final_mask, mask[0])

    if do_clean_mask:
        final_mask = clean_mask(road_lines, final_mask, offset_distance - 10) #TODO: eventualmente aggiungere un parametro per l'additional_cleaning       
    
    return final_mask[np.newaxis, :], np.array(final_pt_coords4Sam), np.array(final_labels4Sam)
    
def clean_mask(road_lines: Union[LineString, List[LineString]],
               final_mask_2d: np.array,
               offset_distance,
               additional_cleaning = False):
    """
    Clean the mask by removing parts outside the offset lines.
    The additional_cleaning parameter is used to remove small holes and small objects from the mask.
    """

    if not isinstance(road_lines, list):
        road_lines = [road_lines]

    line_buffers = [line.buffer(offset_distance) for line in road_lines]
    buffer_roads = rasterize(line_buffers, out_shape=final_mask_2d.shape)
    clear_mask = np.logical_and(final_mask_2d, buffer_roads)
    
    if additional_cleaning: #TODO: controllare meglio cosa fanno queste funzioni e tunare i parametri
        clear_mask = morphology.remove_small_holes(clear_mask, area_threshold=500)
        clear_mask = morphology.remove_small_objects(clear_mask, min_size=500)
        clear_mask = morphology.binary_opening(clear_mask)
        clear_mask = morphology.binary_closing(clear_mask)
    
    return clear_mask
