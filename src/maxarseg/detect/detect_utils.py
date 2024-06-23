import torch
from typing import Union, List
from torchvision.ops import box_convert
import torchvision
import numpy as np
import groundingdino.datasets.transforms as T
from PIL import Image



def GDboxes2SamBoxes(boxes: torch.Tensor, img_shape: Union[tuple[float, float], float]):
    """
    Convert the boxes from the format cxcywh to the format xyxy.
    Inputs:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format cxcywh (the output of GroundingDINO).
        img_shape: tuple (h, w)
        img_res: float, the resolution of the image (mt/pxl).
    Output:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format xyxy.
    """
    if isinstance(img_shape, (float, int)):
        img_shape = (img_shape, img_shape)
    
    h, w =  img_shape
    SAM_boxes = boxes.clone()
    SAM_boxes = SAM_boxes * torch.Tensor([w, h, w, h])
    SAM_boxes = box_convert(boxes=SAM_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return SAM_boxes

def GD_img_load(np_img_rgb: np.array)-> torch.Tensor:
    """
    Transform the image from np.array to torch.Tensor and normalize it.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(np_img_rgb)
    # try:
    #     image_pillow = Image.fromarray(np_img_rgb)
    # except:
    #     print("Error in Image.fromarray")
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed

def filter_on_box_area_mt2(boxes, img_shape: Union[tuple[float, float], float] = None, img_res = None, min_area_mt2 = 0, max_area_mt2 = 1500, box_format = 'cxcywh'):
    """
    Filter boxes based on min and max area.
    Inputs:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format cxcywh (the output of GroundingDINO) or xyxy (rel coords).
        img_shape: tuple (h, w)
        img_res: float, the resolution of the image (mt/pxl).
        min_area_mt2: float
        max_area_mt2: float
        box_format: str, the format of the boxes. 'cxcywh' or 'xyxy'.
    Output:
        keep_ix: torch.Tensor of shape (N,)
    """
    if box_format == 'cxcywh':
        if isinstance(img_shape, (float, int)):
            img_shape = (img_shape, img_shape)
        
        h, w =  img_shape
        tmp_boxes = boxes.clone()
        tmp_boxes = tmp_boxes * torch.Tensor([w, h, w, h])

        area_mt2 = torch.prod(tmp_boxes[:,2:], dim=1) * img_res**2
    
    elif box_format == 'xyxy':
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        area_mt2 = width * height
                
    keep_ix = (area_mt2 > min_area_mt2) & (area_mt2 < max_area_mt2)
    
    return keep_ix

def filter_on_box_ratio(boxes, min_edges_ratio = 0, box_format = 'cxcywh',):
    """
    Filter boxes based on the ratio between the edges.
    """
    if box_format == 'cxcywh':
        keep_ix = (boxes[:,2] / boxes[:,3] > min_edges_ratio) & (boxes[:,3] / boxes[:,2] > min_edges_ratio)
    elif box_format == 'xyxy':
        width = boxes[:, 2] - boxes[:, 0] #xmax - xmin
        height = boxes[:, 3] - boxes[:, 1] #ymax - ymin
        keep_ix = (width / height > min_edges_ratio) & (height / width > min_edges_ratio)
    return keep_ix

def reduce_tree_boxes(boxes, reduce_perc):
    """
    Reduce the size of the boxes by reduce_perc. Keeping the center fixed.
    Input:
        boxes: torch.Tensor of shape (N, 4). Where boxes are in the format cxcywh (the output of GroundingDINO).
        reduce_perc: float, the float to reduce the boxes.
    Output:
        boxes: torch.Tensor of shape (N, 4). Where reduced boxes are in the format cxcywh.
    """
    reduced_boxes = boxes.clone()
    reduced_boxes[:,2:] = reduced_boxes[:,2:] * (1 - reduce_perc)
    return reduced_boxes

def rel2glb_xyxy(rel_xyxy_tree_boxes, top_left_xy):
    """
    Convert the relative coordinates of the boxes to global coordinates.
    Inputs:
        rel_xyxy_tree_boxes: torch.Tensor of shape (N, 4). Where boxes are in the format xyxy.
        top_left_xy: tuple (x, y)
    Output:
        glb_xyxy_tree_boxes: torch.Tensor of shape (N, 4). Where boxes are in the format xyxy.
    """
    glb_xyxy_tree_boxes = rel_xyxy_tree_boxes.clone()
    glb_xyxy_tree_boxes[:,[0,2]] += top_left_xy[0]
    glb_xyxy_tree_boxes[:,[1,3]] += top_left_xy[1]
    return glb_xyxy_tree_boxes