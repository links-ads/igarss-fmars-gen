import torch
import numpy as np

def segment_from_boxes(predictor, boxes, img4Sam, use_bbox = True, use_center_points = False):
    """
    Segment the buildings in the image using the predictor passed as input.
    The image has to be encoded the image before calling this function.
    Inputs:
        predictor: the predictor to use for the segmentation
        building_boxes: a list of tuples or a 2d np.array (b, 4) containing the building's bounding boxes. A single box is in the format (minx, miny, maxx, maxy) = (top left corner, bottom right corner)
        img4Sam: the image previously encoded
        use_bbox: if True, the bounding boxes are used for the segmentation
        use_center_points: if True, the center points of the bounding boxes are used for the segmentation

    Returns:
        mask: a np array of shape (1, h, w). The mask is True where there is a building, False elsewhere
        bboxes: a list of tuples containing the bounding boxes of the buildings used for the segmentation
        #!used_points: a np array of shape (n, 2) where n is the number of buildings. The array contains the center points of the bounding boxes of the buildings in the image
    """

    boxes_t = torch.tensor(boxes, device=predictor.device)
    
    #if use_bboxes
    transformed_boxes = None
    if use_bbox:
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_t, img4Sam.shape[:2])
    
    #if use points
    transformed_points = None
    transformed_point_labels = None
    if use_center_points: #TODO: aggiustare l'utilizzo di punti, al momeno non funziona
        point_coords = torch.tensor([[(sublist[0] + sublist[2])/2, (sublist[1] + sublist[3])/2] for sublist in building_boxes_t], device=predictor.device)
        point_labels = torch.tensor([1] * point_coords.shape[0], device=predictor.device)[:, None]
        transformed_points = predictor.transform.apply_coords_torch(point_coords, img4Sam.shape[:2]).unsqueeze(1)
        transformed_point_labels = point_labels[:, None]

    masks, _, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=transformed_point_labels,
                boxes=transformed_boxes,
                multimask_output=False,
            )
    #mask è un tensore di dimensione (n, 1, h, w) dove n è il numero di maschere (=numero di box passate)
    mask = np.any(masks.cpu().numpy(), axis = 0)

    used_boxes = None
    if use_bbox:
        used_boxes = boxes

    used_points = None
    if use_center_points:
        used_points = point_coords.cpu().numpy()

    return mask, used_boxes, used_points #returna tutti np array
