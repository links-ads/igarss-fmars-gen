import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from typing import Union, List
import numpy as np

def show_mask(mask: np.array, ax = None, rgb_color=[30, 144, 255], alpha = 0.6, random_color = False):
    """
    Take a mask that is a 2D array and show it on the axis ax
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_Linestrings(lines: Union[List[LineString], LineString], ax, color = 'red', linewidth = 1):
    """
    Plots a single or a list of shapely linestrings
    """
    
    if not isinstance(lines, list):
        lines = [lines]

    for line in lines:
        x_s, y_s = line.coords.xy
        ax.plot(x_s, y_s, color=color, linewidth=linewidth)

def show_box(boxes, ax, color='r', lw = 0.5):
    """
    Plot a single or list of boxes. Where the single box is in the format [x0, y0, x1, y1]
    """
    if not isinstance(boxes, list) and not isinstance(boxes, np.ndarray):
        boxes = [boxes]
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=lw))

def show_points(coords: np.array, labels: np.array, ax, marker_size=75):
    """
    Plot an array of points.
    Inputs:
        coords: a np array of shape (n, 2) containing the coordinates of the points
        labels: a np array of shape (n,) containing the labels of the points
        ax: the axis on which to plot the points
        marker_size: the size of the markers
    """
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], color='b', marker='.', s=marker_size, edgecolor='white', linewidth=0.25)
    else:
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='.', s=marker_size, edgecolor='white', linewidth=0.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=0.25)

def plot_comparison(img, masks, alpha = 0.6):
    """
    Plot a comparison between the original image and the image with the masks.
    Inputs:
        img: the original image (np.array of dim (h, w, 3))
        masks: a np.array of masks (dim: (#masks, h, w) or (h, w))
        alpha: the opacity of the masks
    """
    if len(masks.shape) != 3:
        masks = np.expand_dims(masks, axis = 0)
    
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)

    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(img)
    for i in range(masks.shape[0]):
        show_mask(masks[i], ax2, random_color=True)

    #ax2.set_xlim([0, img.shape[1]])
    #ax2.set_ylim([img.shape[0], 0])

    ax1.axis('off')
    ax2.axis('off')
    
def show_img(img, ax = None):
    """
    Show an image on the axis ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')