import numpy as np
from typing import List, Tuple, Union
from skimage import morphology
from shapely.geometry import Polygon
from scipy.signal.windows import tukey


def get_input_pts_and_lbs(tree_boxes_b: List, #list of array of shape (query_img_x, 4)
                          building_boxes_b: List, 
                          max_detect: int):
    input_lbs = []
    input_pts = []
    pts_pad_value = -10
    lbs_pad_value = 0
    for tree_detec, build_detec in zip(tree_boxes_b, building_boxes_b):
        tree_build_detect = np.concatenate((tree_detec, build_detec)) #(query_img_x, 4)
        num_query_img_x = tree_build_detect.shape[0]
        lbs = np.array([[2,3]] * num_query_img_x).reshape(-1,2) #(query_img_x, 2)

        pad_len = max_detect - num_query_img_x
        pad_width = ((0,pad_len),(0, 0))
        padded_tree_build_detect = np.pad(tree_build_detect, pad_width, constant_values=pts_pad_value)
        img_input_pts = np.expand_dims(padded_tree_build_detect, axis = 0).reshape(-1,2,2) # (max_queries, 2, 2)
        input_pts.append(img_input_pts)

        padded_lbs = np.pad(lbs, pad_width, constant_values = lbs_pad_value)# (max_queries, 2)
        input_lbs.append(padded_lbs)

    return np.array(input_pts), np.array(input_lbs) # (batch_size, max_queries, 2, 2), (batch_size, max_queries, 2)


def discern(all_mask_b: np.array, num_trees4img:np.array, num_build4img: np.array):
    """
    Discern the masks of the trees, buildings and padding from the all_mask_b array
    Inputs:
        all_mask_b: np.array of shape (b, masks, h, w)
        num_trees4img: np.array of shape (b,)
        num_build4img: np.array of shape (b,)
    
    Outputs:
        tree_mask_b: np.array of shape (b, h, w)
        build_mask_b: np.array of shape (b, h, w)
        pad_mask_b: np.array of shape (b, h, w)
    """
    h, w = all_mask_b.shape[2:]
    tree_mask_b = build_mask_b = pad_mask_b = np.full((1, h, w), False)

    all_mask_b = np.greater_equal(all_mask_b, 0) #from logits to bool
    
    for all_mask, tree_ix, build_ix in zip (all_mask_b, num_trees4img, num_build4img):
        #all_mask.shape = (num_mask, h, w)
        tree_mask = all_mask[ : tree_ix].any(axis=0) #(h, w)
        tree_mask_b = np.concatenate((tree_mask_b, tree_mask[None, ...]), axis=0) #(b, h, w)

        build_mask = all_mask[tree_ix : (tree_ix + build_ix)].any(axis=0)
        build_mask_b = np.concatenate((build_mask_b, build_mask[None, ...]), axis=0)
        
        pad_mask = all_mask[(tree_ix + build_ix) : ].any(axis=0)
        pad_mask_b = np.concatenate((pad_mask_b, pad_mask[None, ...]), axis=0)
    
    return tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:] #all (b, h, w), slice out the first element

def discern_mode(all_mask_b: np.array, num_trees4img:np.array, num_build4img: np.array, mode: str = 'bchw'):
    """
    Discern the masks of the trees, buildings and padding from the all_mask_b array
    Inputs:
        all_mask_b: np.array of shape (b, masks, h, w)
        num_trees4img: np.array of shape (b,)
        num_build4img: np.array of shape (b,)
        mode: 'bchw' or 'cbhw'. To specify the output dimension. [batch channel height width] or [channel batch height width]
    
    Outputs:
        out: np.array of shape (b, c, h, w) or (c, b, h, w)
    """
    h, w = all_mask_b.shape[2:]
    tree_mask_b = build_mask_b = pad_mask_b = np.full((1, h, w), False)

    all_mask_b = np.greater_equal(all_mask_b, 0) #from logits to bool
    
    
    
    for all_mask, tree_ix, build_ix in zip (all_mask_b, num_trees4img, num_build4img):
        #all_mask.shape = (num_mask, h, w)
        tree_mask = all_mask[ : tree_ix].any(axis=0) # Squash the tree masks. Get shape (h, w)
        tree_mask_b = np.concatenate((tree_mask_b, tree_mask[None, ...]), axis=0) #(b, h, w)

        build_mask = all_mask[tree_ix : (tree_ix + build_ix)].any(axis=0) # Squash the build masks. Get shape (h, w)
        build_mask_b = np.concatenate((build_mask_b, build_mask[None, ...]), axis=0)
        
        pad_mask = all_mask[(tree_ix + build_ix) : ].any(axis=0)
        pad_mask_b = np.concatenate((pad_mask_b, pad_mask[None, ...]), axis=0)
    
    if mode == 'bchw':
        out = np.stack((tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:]), axis=1) # (b, c, h, w) , slice out the first element of dim 1
    elif mode == 'cbhw':
        out = np.stack((tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:]), axis=0) # (c, b, h, w) , slice out the first element of dim 1
    return out

def discern_mode_smooth(all_mask_b: np.array, num_trees4img:np.array, num_build4img: np.array, mode: str = 'bchw'):
    """
    Discern the masks of the trees, buildings and padding from the all_mask_b array
    Inputs:
        all_mask_b: np.array of shape (b, masks, h, w)
        num_trees4img: np.array of shape (b,)
        num_build4img: np.array of shape (b,)
        mode: 'bchw' or 'cbhw'. To specify the output dimension. [batch channel height width] or [channel batch height width]
    
    Outputs:
        out: np.array of shape (b, c, h, w) or (c, b, h, w)
    """
    h, w = all_mask_b.shape[2:]
    tree_mask_b = build_mask_b = pad_mask_b = np.full((1, h, w), float('-inf'), dtype=np.float32)
    
    for all_mask, tree_ix, build_ix in zip (all_mask_b, num_trees4img, num_build4img):
        #all_mask.shape = (num_mask, h, w)
        tree_mask = all_mask[ : tree_ix].max(axis=0, initial = float('-inf')) # Squash the tree masks. Get shape (h, w)
        tree_mask_b = np.concatenate((tree_mask_b, tree_mask[None, ...]), axis=0) #(b, h, w)

        build_mask = all_mask[tree_ix : (tree_ix + build_ix)].max(axis=0, initial = float('-inf')) # Squash the build masks. Get shape (h, w)
        build_mask_b = np.concatenate((build_mask_b, build_mask[None, ...]), axis=0)
         
        pad_mask = all_mask[(tree_ix + build_ix) : ].max(axis=0, initial = float('-inf'))
        pad_mask_b = np.concatenate((pad_mask_b, pad_mask[None, ...]), axis=0)
    
    if mode == 'bchw':
        out = np.stack((tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:]), axis=1) # (b, c, h, w) , slice out the first element of dim 1
    elif mode == 'cbhw':
        out = np.stack((tree_mask_b[1:], build_mask_b[1:], pad_mask_b[1:]), axis=0) # (c, b, h, w) , slice out the first element of dim 1
    return out

def rmv_mask_b_overlap(overlapping_masks_b: np.array): #(b, c, h, w)
    """
    Remove overlapping between the masks. Giving priority according to the inverse of the order of
    the masks.
    Third (building) mask has priority over second (trees) mask, and so on.
    """

    disjoined_masks_b = np.copy(overlapping_masks_b)
    for i in range(overlapping_masks_b.shape[1] - 1):
        sum_mask = np.sum(overlapping_masks_b[:,i:], axis=1)
        disjoined_masks_b[:,i] = np.where(sum_mask > 1, False, overlapping_masks_b[:, i])
    
    return disjoined_masks_b

#Use the batch version of this function
def rmv_mask_overlap(overlapping_masks: np.array):
    """
    Remove overlapping between the masks. Giving priority according to the inverse of the order of
    the masks.
    Third (building) mask has priority over second (trees) mask, and so on.
    Inputs:
        overlapping_masks: np.array of shape (c, h, w)
    Outputs:
        no_overlap_masks: np.array of shape (c, h, w)
    """
    no_overlap_masks = np.copy(overlapping_masks)
    for i in range(overlapping_masks.shape[0] - 1):
        sum_mask = np.sum(overlapping_masks[i:], axis=0)
        no_overlap_masks[i] = np.where(sum_mask > 1, False, overlapping_masks[i])

    return no_overlap_masks

def write_canvas(canvas: np.array,
                 patch_masks_b: np.array,
                 img_ixs: np.array,
                 stride: int,
                 total_cols: int) -> np.array:
    """
    Write the patch masks in the canvas
    Inputs:
        canvas: np.array of shape (channel, h_tile, w_tile)
        patch_masks_b: np.array of shape (b, channel, h_patch, w_patch)
        img_ixs: np.array of shape (b,)
    """
    size = patch_masks_b.shape[-1]
    #print("img_ixs", img_ixs)
    for img_ix, patch_mask in zip(img_ixs, patch_masks_b):
        rows_changed = img_ix // total_cols
        cols_changed = img_ix % total_cols
        inv_base = (canvas.shape[1] - 1 - size) - (stride * rows_changed)
        base = (stride * cols_changed)
        canva_writable_space = canvas[:, inv_base: inv_base + size, base: base + size].shape[1:] #useful when reached the border of the canva
        #print('\nparte di canva', canvas[:, inv_base: inv_base + size, base: base + size].shape)
        #print('patch', patch_mask[:, :canva_writable_space[0], :canva_writable_space[1]].shape)
        canvas[:, inv_base: inv_base + size, base: base + size] = patch_mask[:, :canva_writable_space[0], :canva_writable_space[1]]

    return canvas

def write_canvas_geo(canvas: np.array,
                    patch_masks_b: np.array,
                    top_lft_indexes: List,
                    smooth: bool) -> np.array:
    """
    Write the patch masks in the canvas.

    Args:
        canvas (np.array): The canvas to write the patch masks on. It should have shape (channel, h_tile, w_tile).
        patch_masks_b (np.array): The patch masks to be written on the canvas. It should have shape (b, channel, h_patch, w_patch).
        top_lft_indexes (List): The top left indexes of each patch mask in the canvas.
        smooth (bool): If True, it expects patch_mask to have logits, otherwise it should contain bools.

    Returns:
        np.array: The updated canvas with the patch masks written on it.
    """
    
    size = patch_masks_b.shape[-1]
    for patch_mask, top_left_index in zip(patch_masks_b, top_lft_indexes):
        I = np.s_[:, top_left_index[0]: top_left_index[0] + size, top_left_index[1]: top_left_index[1] + size] #index var in the canvas where to add the patch
        #max_idxs is useful when reached the border of the canva, it contains the height and width that you can write on the canva
        max_idxs = canvas[I].shape[1:]
        
        #print('\nparte di canva', canvas[:, inv_base: inv_base + size, base: base + size].shape)
        #print('patch', patch_mask[:, :max_idxs[0], :max_idxs[1]].shape)
        if smooth:
            canvas[I] = np.maximum(canvas[I], patch_mask[:, :max_idxs[0], :max_idxs[1]]) #element-wise max between the canva and the patch
        #elif smooth == 'avg':
        #    canvas[I] = (canvas[I] + patch_mask[:, :max_idxs[0], :max_idxs[1]]) / 2 #TODO: this is wrong
        else:
            canvas[I] = patch_mask[:, :max_idxs[0], :max_idxs[1]]

    return canvas 


def write_canvas_geo_window(canvas: np.ndarray,
                            weights: np.ndarray,
                    patch_masks_b: np.ndarray,
                    top_lft_indexes: List):
    """
    Write the patch masks in the canvas.

    Args:
        canvas (np.array): The canvas to write the patch masks on. It should have shape (channel, h_tile, w_tile).
        weights (np.array): The canvas that contains the weights used to weight the contribution of each patch. It should have shape (channel, h_tile, w_tile).
        patch_masks_b (np.array): The patch masks to be written on the canvas. It should have shape (b, channel, h_patch, w_patch).
        top_lft_indexes (List): The top left indexes of each patch mask in the canvas.
        smooth (bool): If True, it expects patch_mask to have logits, otherwise it should contain bools.

    Returns:
        np.array: The updated canvas with the patch masks written on it.
    """
    # initialize the window, a 2d array containing a cosine window, of the same size as the patch
    # window = np.outer(np.hanning(patch_masks_b.shape[-1]), np.hanning(patch_masks_b.shape[-1]))

    # tukey window
    window1d = tukey(patch_masks_b.shape[-1], alpha=0.5) # TODO: make it conditional
    window = np.outer(window1d, window1d)
    
    mask_size = patch_masks_b.shape[-1]
    
    for patch_mask, top_left_index in zip(patch_masks_b, top_lft_indexes):
        c_start_y = max(0, top_left_index[0])
        c_start_x = max(0, top_left_index[1])
        c_end_y = min(canvas.shape[1], top_left_index[0] + mask_size)
        c_end_x = min(canvas.shape[2], top_left_index[1] + mask_size)
        
        m_start_y = max(0, -top_left_index[0])
        m_start_x = max(0, -top_left_index[1])
        m_end_y = m_start_y + (c_end_y - c_start_y)
        m_end_x = m_start_x + (c_end_x - c_start_x)

        # if m_start_y != 0 or m_start_x != 0 or m_end_y != mask_size or m_end_x != mask_size:
        #     pass
                
        canvas[0, c_start_y: c_end_y, c_start_x: c_end_x] = canvas[0, c_start_y: c_end_y, c_start_x: c_end_x] + patch_mask[0, m_start_y: m_end_y, m_start_x: m_end_x] * window[m_start_y: m_end_y, m_start_x: m_end_x]
        canvas[1, c_start_y: c_end_y, c_start_x: c_end_x] = canvas[1, c_start_y: c_end_y, c_start_x: c_end_x] + patch_mask[1, m_start_y: m_end_y, m_start_x: m_end_x] * window[m_start_y: m_end_y, m_start_x: m_end_x]

        weights[c_start_y: c_end_y, c_start_x: c_end_x] = weights[c_start_y: c_end_y, c_start_x: c_end_x] + window[m_start_y: m_end_y, m_start_x: m_end_x]

    return canvas, weights



    
def clean_masks(masks: np.ndarray, area_threshold = 80, min_size = 80) -> np.ndarray:
    """
    Cleans the input masks by removing small holes and objects.

    Args:
        masks (np.array): The input masks to be cleaned. Can be a single mask or a stack of masks.
        area_threshold (int, optional): The area threshold for removing small holes. Defaults to 80.
        min_size (int, optional): The minimum size for removing small objects. Defaults to 80.

    Returns:
        np.array: The cleaned masks. With the same dimensions as the input masks.
    """
    single_mask = False
    if len(masks.shape) == 2:
        single_mask = True
        masks = np.expand_dims(masks, axis=0)
    
    clear_masks = []
    masks_int = masks.astype(np.uint8)
    for mask in masks_int:
        clear_mask = morphology.remove_small_holes(mask, area_threshold = area_threshold)
        clear_mask = morphology.remove_small_objects(clear_mask, min_size = min_size)
        #clear_mask = morphology.binary_opening(mask)
        #clear_mask = morphology.binary_closing(clear_mask)
        clear_masks.append(clear_mask)

    if single_mask:
        clear_masks = clear_masks[0]
    else:
        clear_masks = np.stack(clear_masks, axis=0)
        
    return clear_masks

def merge_masks(masks: np.ndarray):
    """
    Merges multiple masks into a single mask.
    Labels:
        road = 0
        tree = 1
        building = 2
        background = 255

    Args:
        masks (list): A list of masks to be merged.

    Returns:
        np.ndarray: The merged mask.
    """
    merged_mask = np.full_like(masks[0], fill_value=255, dtype=np.uint8)
    for i, mask in enumerate(masks):
        merged_mask[mask.astype(bool)] = i

    return merged_mask
