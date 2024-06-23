import numpy as np
import torch


def ESAM_from_inputs(original_img_tsr: torch.tensor, #b, c, h, w
                    input_points: torch.tensor, #b, max_queries, 2, 2
                    input_labels: torch.tensor, #b, max_queries, 2
                    efficient_sam,
                    num_parall_queries: int = 50,
                    device = 'cpu',
                    empty_cuda_cache = True):
    
    img_b_tsr = original_img_tsr.div(255)
    batch_size, _, input_h, input_w = img_b_tsr.shape
    
    img_b_tsr = img_b_tsr.to(device)
    input_points = input_points.to(device)
    input_labels = input_labels.to(device)

    image_embeddings = efficient_sam.get_image_embeddings(img_b_tsr)
    
    stop = input_points.shape[1]
    if stop > 0: #if there is at least a query in a single image in the batch
        for i in range(0, stop , num_parall_queries):
            start_idx = i
            end_idx = min(i + num_parall_queries, stop)
            #TODO: check if multimask_output False is faster
            predicted_logits, predicted_iou = efficient_sam.predict_masks(image_embeddings,
                                                                    input_points[:, start_idx: end_idx],
                                                                    input_labels[:, start_idx: end_idx],
                                                                    multimask_output=True,
                                                                    input_h = input_h,
                                                                    input_w = input_w,
                                                                    output_h=input_h,
                                                                    output_w=input_w)
            
            if i == 0:
                #print('predicetd_logits:', predicted_logits.shape)
                np_complete_masks = predicted_logits[:,:,0].cpu().detach().numpy()
            else:
                np_complete_masks = np.concatenate((np_complete_masks, predicted_logits[:,:,0].cpu().detach().numpy()), axis=1)
            #TODO: check if empty_cuda_cache False is faster
            if empty_cuda_cache:
                del predicted_logits, predicted_iou
                torch.cuda.empty_cache()
    else: #if there are no queries (in any image in the batch)
        np_complete_masks = np.ones((batch_size, 0, input_h, input_w)) * float('-inf') #equal to set False on all the mask
        
    return np_complete_masks #shape (b, masks, h, w)

def ESAM_from_inputs_fast(original_img_tsr: torch.Tensor, #b, c, h, w
                        input_points: torch.Tensor, #b, max_queries, 2, 2
                        input_labels: torch.Tensor, #b, max_queries, 2
                        efficient_sam,
                        num_tree_boxes, #(b, 1)
                        num_parall_queries: int = 5,
                        device = 'cpu'):
    
    num_tree_boxes = int(num_tree_boxes[0])
    
    original_img_tsr = original_img_tsr.div(255)
    batch_size, _, input_h, input_w = original_img_tsr.shape
    
    original_img_tsr = original_img_tsr.to(device)
    input_points = input_points.to(device)
    input_labels = input_labels.to(device)
    with torch.no_grad():
        image_embeddings = efficient_sam.get_image_embeddings(original_img_tsr)

    tree_build_mask = torch.full((2, input_h, input_w), float('-inf'), dtype = torch.float32, device = device)
    num_batch_tree_only = num_tree_boxes // num_parall_queries
    trees_in_mixed_batch = round(num_parall_queries * (num_tree_boxes/num_parall_queries -  num_tree_boxes // num_parall_queries))

    stop = input_points.shape[1]

    for y, i in enumerate(range(0, stop , num_parall_queries)):
        start_idx = i
        end_idx = min(i + num_parall_queries, stop)

        with torch.no_grad():
            predicted_logits, predicted_iou = efficient_sam.predict_masks(image_embeddings,
                                                                    input_points[:, start_idx: end_idx],
                                                                    input_labels[:, start_idx: end_idx],
                                                                    multimask_output=True,
                                                                    input_h = input_h,
                                                                    input_w = input_w,
                                                                    output_h=input_h,
                                                                    output_w=input_w)
        
        masks = predicted_logits[0,:,0]#.cpu().detach().numpy() # (num_img, prompt, multi, h, w) -> (max_queries, h, w)

        
        # poly
        # append to geodataframe


        if y < num_batch_tree_only or input_points[0, start_idx: end_idx].shape[0] == trees_in_mixed_batch: #only trees
            tree_build_mask[0] = torch.max(tree_build_mask[0], torch.max(masks, dim=0).values)
        elif y > num_batch_tree_only or trees_in_mixed_batch == 0: #only build
            tree_build_mask[1] = torch.max(tree_build_mask[1], torch.max(masks, dim=0).values)
        else: #trees and build
            tree_build_mask[0] = torch.max(tree_build_mask[0], torch.max(masks[:trees_in_mixed_batch], dim=0).values)
            tree_build_mask[1] = torch.max(tree_build_mask[1], torch.max(masks[trees_in_mixed_batch:], dim=0).values)

    # filter the -inf values to 0 TODO: move outside
    tree_build_mask[0] = torch.where(tree_build_mask[0] == float('-inf'), torch.tensor(0, dtype = torch.float32, device = device), tree_build_mask[0])
    tree_build_mask[1] = torch.where(tree_build_mask[1] == float('-inf'), torch.tensor(0, dtype = torch.float32, device = device), tree_build_mask[1])

    tree_build_mask = tree_build_mask.cpu().detach().numpy()
    return tree_build_mask #shape (b, masks, h, w)