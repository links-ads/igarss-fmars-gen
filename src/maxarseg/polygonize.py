import rasterio as rio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib import collections  as mc


def onehot(components: np.ndarray) -> np.ndarray:
    oh = np.zeros((components.max() + 1, *components.shape), dtype=np.uint8)
    for i in range(oh.shape[0]):
        oh[i][components == i] = 1
    if 0 in np.unique(components):
        oh = oh[1:]
    return oh


def apply_transform(polygon: Polygon, transform: rio.Affine) -> Polygon:
    return Polygon([transform * c for c in polygon.exterior.coords])


def polygonize_raster(raster: np.ndarray, tolerance: float = 0.1, transform: rio.transform.Affine = None, crs: str = None, pixel_threshold: int = 100) -> gpd.GeoDataFrame:
    data = defaultdict(list)
    onehot_raster = onehot(raster)
    for i in range(onehot_raster.shape[0]):
        mask = onehot_raster[i]
        if mask.sum() < pixel_threshold:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, tolerance * perimeter, True)
            contour = approx.squeeze()
            if contour.shape[0] < 3:
                continue
            poly = shape({"type": "Polygon", "coordinates": [contour]})
            if transform is not None:
                poly = apply_transform(poly, transform)
            data["geometry"].append(poly)
            data["component"].append(i)
    return gpd.GeoDataFrame(data, crs=crs)


# version of polygonize_raster, which returns the gdf with an additional field, containing the value of the pixels contained in each polygon
def polygonize_with_values(raster: np.ndarray, class_id: int, tolerance: float = 0.1, transform: rio.transform.Affine = None, crs: str = None, pixel_threshold: int = 100) -> gpd.GeoDataFrame:
    # pixel_threshold is the minimum number of pixels that a polygon must contain to be considered
    data = defaultdict(list)
    onehot_raster = onehot(raster)
    for i in range(onehot_raster.shape[0]):
        mask = onehot_raster[i]
        if mask.sum() < pixel_threshold:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, tolerance * perimeter, True)
            contour = approx.squeeze()
            if contour.shape[0] < 3:
                continue
            poly = shape({"type": "Polygon", "coordinates": [contour]})
            if transform is not None:
                poly = apply_transform(poly, transform)
            data["geometry"].append(poly)
            data["component"].append(i)
            data["class_id"].append(class_id)
            # if(raster[contour[:, 1], contour[:, 0]].mean() > 1):
            #     print(raster[contour[:, 1], contour[:, 0]])
            #     print(raster[contour[:, 1], contour[:, 0]].mean())
    return gpd.GeoDataFrame(data, crs=crs)


def angle_between_lines(line1, line2):
    a1, b1, _ = line1
    a2, b2, _ = line2
    angle = np.arctan2(a1*b2 - a2*b1, a1*a2 + b1*b2)
    return angle


def minimum_angle(line1: np.ndarray, line2: np.ndarray) -> float:
    """return the minimum angle between two lines"""
    angle = angle_between_lines(line1, line2)
    abs_angle = np.abs(angle)
    sign = np.sign(angle)
    if abs_angle > np.pi / 2:
        angle = (np.pi - abs_angle) * -sign
    return angle


def line_from_points(point1: tuple, point2: tuple) -> np.ndarray:
    """Given two points, return the line parameters in the Hesse normal form"""
    x1, y1 = point1
    x2, y2 = point2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return np.array([a, b, c])


def rotate_point(p: tuple, center: tuple, angle: float) -> tuple:
    """rotate a point around a center by a given angle"""
    x, y = p
    cx, cy = center
    x -= cx
    y -= cy
    x_new = x * np.cos(angle) - y * np.sin(angle)
    y_new = x * np.sin(angle) + y * np.cos(angle)
    x_new += cx
    y_new += cy
    return x_new, y_new


def middle_point(p1: tuple, p2: tuple) -> tuple:
    """return the middle point between two points"""
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def find_longest_edge(polygon: Polygon) -> Tuple[float, float]:
    max_length = 0
    max_edge = None
    max_index = -1
    for i in range(len(polygon.exterior.coords) - 1):
        a = polygon.exterior.coords[i]
        b = polygon.exterior.coords[i + 1]
        length = np.linalg.norm(np.array(a) - np.array(b))
        if length > max_length:
            max_length = length
            max_edge = (a, b)
            max_index = i
    return max_edge, max_length, max_index


def remove_angles(polygon: Polygon, min_angle: float, max_angle: float) -> Polygon:
    coords = list(polygon.exterior.coords)
    i = 0
    while i < len(coords) - 2:
        a = coords[i]
        b = coords[i + 1]
        c = coords[i + 2]
        l1 = line_from_points(a, b)
        l2 = line_from_points(b, c)
        rads = angle_between_lines(l1, l2)
        angle = abs(np.rad2deg(rads))
        if angle < min_angle or max_angle < angle:
            coords.pop(i + 1)
        else:
            i += 1
    return Polygon(coords)


def rotate_segment(point1, point2, angle):
    # Find the middle point
    x1, y1 = point1
    x2, y2 = point2
    middle_x = (x1 + x2) / 2
    middle_y = (y1 + y2) / 2
    # Convert angle to radians
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Rotate the edge by a specific angle around the middle point
    new_x1 = (x1 - middle_x) * cos_angle - (y1 - middle_y) * sin_angle + middle_x
    new_y1 = (x1 - middle_x) * sin_angle + (y1 - middle_y) * cos_angle + middle_y
    new_x2 = (x2 - middle_x) * cos_angle - (y2 - middle_y) * sin_angle + middle_x
    new_y2 = (x2 - middle_x) * sin_angle + (y2 - middle_y) * cos_angle + middle_y

    # Return the new coordinates of the rotated edge
    return ((new_x1, new_y1), (new_x2, new_y2))


def are_lines_parallel(line1, line2, tolerance = 1e-3):
    angle = angle_between_lines(line1, line2)
    return abs(angle) < tolerance or abs(angle - np.pi) < tolerance


def parallel_line(line, point):
    a, b, c = line
    x0, y0 = point
    c_parallel = -a*x0 - b*y0
    parallel = np.array([a, b, c_parallel])
    return parallel


def longest_segment(coords: List[Tuple[float, float]]) -> Tuple[tuple, float]:
    """Returns the longest segment in a list of coordinates"""
    max_length = 0
    max_segment = None
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        length = np.linalg.norm(np.array(a) - np.array(b))
        if length > max_length:
            max_length = length
            max_segment = (a, b)
    return max_segment, max_length


def align_polygon(polygon: Polygon) -> Polygon:
    """
    Aligns the each edge of the polygon to the closest direction among the two
    orientations of its minimum rotated rectangle.
    """
    min_rect = polygon.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)
    a, b, c = rect_coords[0], rect_coords[1], rect_coords[2]

    l1 = line_from_points(rect_coords[0], rect_coords[1])
    l2 = line_from_points(rect_coords[1], rect_coords[2])

    # iterate over polygon edges
    new_coords = []
    coords = list(polygon.exterior.coords)
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]

        # Find the angle between the segment and the line
        l = line_from_points(a, b)
        angle_1 = minimum_angle(l, l1)
        angle_2 = minimum_angle(l, l2)

        if abs(angle_1) < abs(angle_2):
            angle = angle_1
        else:
            angle = angle_2
        # adjust the edge
        ar, br = rotate_segment(a, b, angle=angle)
        new_coords.append([ar, br])
    return new_coords


def plot_line(line: np.ndarray, xlim: tuple, ylim: tuple, **kwargs):
    a, b, c = line
    x1, x2 = xlim
    y1, y2 = ylim
    if abs(b) < 1e-3:
        x = -c/a
        plt.axvline(x, color='r')
    elif abs(a) < 1e-3:
        y = -c/b
        plt.axhline(y, color='r')
    else:
        x = np.linspace(x1, x2, 50)
        y = (-c - a*x) / b
        # filter out points outside the plot
        #values = [(x, y) for x, y in zip(x, y) if x1 <= x <= x2 and y1 <= y <= y2]
        indices = [i for i in range(len(x)) if x1 <= x[i] <= x2 and y1 <= y[i] <= y2]
        plt.plot(x[indices], y[indices], **kwargs)


def format_point(point: tuple) -> str:
    return f"{point[0]:.2f},{point[1]:.2f}"


def orthogonal_line(line, point):
    a, b, c = line
    x0, y0 = point
    a_ortho, b_ortho = -b, a
    c_ortho = -a_ortho*x0 - b_ortho*y0
    ortho = np.array([a_ortho, b_ortho, c_ortho])
    return ortho


def filter_segments(coords: List[tuple], centroid: tuple, threshold: float = 0.1) -> List[tuple]:
    result = []
    _, max_length = longest_segment(coords)

    print(f"max length: {max_length * threshold}")
    plt.figure(figsize=(15, 15))
    # plt.xlim(100, 500)
    # plt.ylim(100, 500)

    for i in range(len(coords)):
        a, b = coords[i]
        c, d = coords[(i + 1) % len(coords)]
        l1 = line_from_points(a, b)
        l2 = line_from_points(c, d)

        # plot_line(l1, start=min(a[0] ,b[0]) - 10, end=max(a[0] ,b[0]) + 10)
        # plot_line(l2, start=min(c[0] ,d[0]) - 10, end=max(c[0] ,d[0]) + 10)
        x, y = zip(*[a, b, c, d])
        plt.scatter(x, y, color="red", marker="x", s=5, zorder=5)
        if are_lines_parallel(l1, l2):
            l1 = orthogonal_line(l1, b)
            e = intersection_point(l1, l2)
            # plot_line(l1, start=min(a[0] ,b[0]) - 10, end=max(a[0] ,b[0]) + 10, color="magenta")
            x, y = zip(*[e])
            plt.scatter(x, y, color="magenta", marker="x", s=150, zorder=10)

            length_a = np.linalg.norm(np.array(a) - np.array(b))
            length_b = np.linalg.norm(np.array(c) - np.array(d))
            length_c = np.linalg.norm(np.array(e) - np.array(b))
            if length_c < threshold * max_length:
                if length_a > length_b:
                    edge = (a, b)
                else:
                    edge = (c, d)
                result.append(edge)
                plt.gca().add_collection(mc.LineCollection([edge], colors="blue", linewidths=2))
                continue

        plt.gca().add_collection(mc.LineCollection([(a, b)], colors="blue", linewidths=2))
        result.append((a, b))

    return result


def perpendicular_distance(point: tuple, line: np.ndarray) -> float:
    """Computes the perpendicular distance between a point and a line."""
    a, b, c = line
    x, y = point
    return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)


def average(points: List[tuple]) -> tuple:
    """Computes the average of a list of points."""
    x = sum([p[0] for p in points]) / len(points)
    y = sum([p[1] for p in points]) / len(points)
    return x, y


def intersection_point(line1: np.ndarray, line2: np.ndarray) -> tuple:
    """Computes the intersection point of two lines, given in Hesse normal form."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-3:
        raise ValueError("Lines are parallel")
    x = (b2*c1 - b1*c2) / det
    y = (a1*c2 - a2*c1) / det
    return abs(x), abs(y)


def filterv2(coords: List[tuple], length_thresh: float, dist_thresh: float) -> List[tuple]:
    result = []
    skip = set()
    _, max_length = longest_segment(coords)
    for i in range(len(coords) - 1):
        if i in skip:
            continue
        a, b = coords[i]
        # discard it immediately if it is too short
        length = np.linalg.norm(np.array(a) - np.array(b))
        if length < length_thresh * max_length:
            continue
        # if the lines are parallel, check if the orthogonal distance is too small
        # if so, compute a middle point and discard the segment
        j = (i + 1) % len(coords)
        should_stop = False
        new_as = list()
        while not should_stop:
            c, d = coords[j]
            l1 = line_from_points(a, b)
            l2 = line_from_points(c, d)
            # if they are not parallel, we can stop
            # we store this to use it later, if we need to stop because of the distance
            are_parallel = are_lines_parallel(l1, l2)
            if not are_parallel:
                should_stop = True
            else:
                # otherwise, check the perpendicular distance
                dist = perpendicular_distance(c, l1)
                if dist < dist_thresh * max_length:
                    a2 = intersection_point(orthogonal_line(l1, a), l2)
                    new_as.append(a2)
                    skip.add(j)
                    j = (j + 1) % len(coords)
                    continue
                else:
                    should_stop = True
            # if we should stop, we can compute new_b and add the segment
            if should_stop and j not in skip:
                l = l2 if not are_parallel else orthogonal_line(l2, c)
                new_a = average(new_as) if len(new_as) > 0 else a
                new_b = intersection_point(l, parallel_line(l1, new_a))
                result.append((new_a, new_b))

    return result


def merge_segments(coords: List[tuple], threshold: float = 0.1) -> List[tuple]:
    result = []
    i  = 0
    j = i
    while i < len(coords):
        a, b = coords[i]
        c, d = coords[(i + 1) % len(coords)]
        if are_lines_parallel(line_from_points(a, b), line_from_points(c, d)):
            length_a = np.linalg.norm(np.array(a) - np.array(b))
            length_b = np.linalg.norm(np.array(c) - np.array(d))
            length_c = np.linalg.norm(np.array(b) - np.array(c))
            if length_c < threshold * (length_a + length_b):
                print("merging")
                mid_c = middle_point(b, c)
                line_a = line_from_points(a, b)
                line_b = line_from_points(c, d)
                lin_c = parallel_line(line_a, mid_c)
                new_a = intersection_point(orthogonal_line(line_a, a), lin_c)
                new_b = intersection_point(orthogonal_line(line_b, d), lin_c)
                result.append((new_a, new_b))
                i += 2
                continue
        result.append((a, b))
        i += 1
    return result


def link_segments(coords: List[tuple]) -> Polygon:
    poly_coords = []
    for i in range(len(coords)):
        a, b = coords[i]
        c, d = coords[(i + 1) % len(coords)]
        l1 = line_from_points(a, b)
        l2 = line_from_points(c, d)

        if are_lines_parallel(l1, l2):
            l1 = orthogonal_line(l1, b)
            poly_coords.append(b)

        poly_coords.append(intersection_point(l1, l2))
    poly_coords.insert(0, poly_coords[-1])
    return Polygon(poly_coords)