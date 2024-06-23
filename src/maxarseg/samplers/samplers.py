from torchgeo.samplers.utils import get_random_bounding_box, tile_to_chips
from torchgeo.samplers.single import RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import GeoDataset, BoundingBox
from maxarseg.samplers.samplers_utils import path_2_tile_aoi, boundingBox_2_Polygon, boundingBox_2_centralPoint
from torchgeo.samplers.constants import Units
from typing import Optional, Union
from collections.abc import Iterator
import torch
import math
from maxarseg.geo_datasets import geoDatasets

class SinglePatchSampler:
    """
    To be used with SingleTileDataset
    Sample a single patch from a dataset.
    """
    def __init__(self, dataset, patch_size, stride):
        self.dataset = dataset
        assert patch_size > 0
        self.patch_size = patch_size #pxl
        if dataset.transform[0] != -dataset.transform[4]:
            raise ValueError("The pixel scale in x and y directions are different.")
        self.patch_size_meters = patch_size * dataset.transform[0]
        
        assert stride > 0
        self.stride = stride #pxl
        if self.stride is None:
            self.stride = self.patch_size
        self.stride_meters = self.stride * dataset.transform[0]
    
    def tile_to_chips(self) -> tuple[int, int]:
        rows = math.ceil((self.dataset.height - self.patch_size) / self.stride) + 1
        cols = math.ceil((self.dataset.width - self.patch_size) / self.stride) + 1
        return rows, cols
        
    def __iter__(self):
        rows, cols = self.tile_to_chips()
        discarder_chips = 0
        for i in range(rows):
            miny = self.dataset.bounds.bottom + i * self.stride_meters
            maxy = miny + self.patch_size_meters

            # For each column...
            for j in range(cols):
                minx = self.dataset.bounds.left + j * self.stride_meters
                maxx = minx + self.patch_size_meters
                selected_bbox = geoDatasets.noTBoundingBox(minx, maxx, miny, maxy)
                selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)
                if self.dataset.tile_aoi_gdf.intersects(selected_bbox_polygon).any():
                    yield (minx, maxx, miny, maxy)
                else:
                    discarder_chips += 1
                    continue
        print('Discarded empty chips: ', discarder_chips)
        print('True num of batch: ', len(self) - discarder_chips)         

    def __len__(self) -> int:
        return self.tile_to_chips()[0]*self.tile_to_chips()[1]

# Samplers per Base Datasets
class MyRandomGeoSampler(RandomGeoSampler):
    """
    Sample a single random bounding box from a dataset (does NOT support batches).
    Check that the random bounding box is inside the tile's polygon.
    """
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        verbose: bool = False
    ) -> None:

        super().__init__(dataset, size, length, roi, units)
        self.verbose = verbose
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        i = 0
        while i < len(self):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]

            tile_path = hit.object
            tile_polyg = path_2_tile_aoi(tile_path)

            bounds = BoundingBox(*hit.bounds) #TODO: ridurre i bounds usando il bbox del geojson
            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)
            #rnd_bbox_polyg = boundingBox_2_Polygon(bounding_box)
            rnd_central_point = boundingBox_2_centralPoint(bounding_box)

            #se il punto centrale della rnd_bbox è nel poligono (definito con geojson) del tile
            if rnd_central_point.intersects(tile_polyg):
                if self.verbose: #TODO: magari in futuro togliere il verbose per velocizzare
                    print('In sampler')
                    print('tile_polyg', tile_polyg)
                    print()
                i += 1
                yield bounding_box
            else:
                continue

class MyGridGeoSampler(GridGeoSampler):
    """
    Sample a single bounding box in a grid fashion from a dataset (does NOT support batches).
    Check that the bounding box is inside the tile's polygon.
    """
    def __init__(self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        ) -> None:

        super().__init__(dataset, size, stride, roi, units)
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits: #These hits are all the tiles that intersect the roi (region of interest). If roi not specified then hits = all the tiles
            tile_path = hit.object
            tile_polygon = path_2_tile_aoi(tile_path)

            #print('In sampler')
            #print('tile_polygon: ', tile_polygon)

            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    selected_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)
                    if selected_bbox_polygon.intersects(tile_polygon):
                        #print("selected_bbox_polygon", selected_bbox_polygon)
                        yield selected_bbox
                    else:
                        continue
    
class WholeTifGridGeoSampler(GridGeoSampler):
    """
    Sample a batch of bounding boxes from a dataset.
    Returns all possible patches even if they are empty.
    """
    def __init__(self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        batch_size: int,
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        ) -> None:

        super().__init__(dataset, size, stride, roi, units)
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        batch = []
        # For each tile...
        for k, hit in enumerate(self.hits): #These hits are all the tiles that intersect the roi (region of interest). If roi not specified then hits = all the tiles
            tile_path = hit.object
            tile_polygon = path_2_tile_aoi(tile_path)

            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt
            
            empty_chips = 0
            valid_chips = 0
            
            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    selected_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    batch.append(selected_bbox)
                    
                    #Check if the selected_bbox intersects the tile_polygon (to avoid all black images)
                    selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)

                    if selected_bbox_polygon.intersects(tile_polygon):
                        valid_chips += 1
                    else:
                        empty_chips += 1
                    
                    is_last_batch = k == len(self.hits) - 1 and i == rows - 1 and j == cols - 1
                    
                    if len(batch) == self.batch_size or is_last_batch:
                        if is_last_batch and len(batch) < self.batch_size:
                            #print('Last batch not full. Only', len(batch), 'chips')
                            #pad the last batch with the last selected_bbox if it is not full
                            batch.extend([selected_bbox] * (self.batch_size - len(batch)))

                        yield batch
                        batch = []
        print('Valid patches: ', valid_chips)
        print('Empty patches: ', empty_chips)

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return math.ceil(self.length / self.batch_size)
    
        
    def get_num_rows_cols(self):
        hit = self.hits[0] #get the first and only tile
        bounds = BoundingBox(*hit.bounds) #get its bounds
        return tile_to_chips(bounds, self.size, self.stride)

class BatchGridGeoSampler(GridGeoSampler):
    """
    Sample a batch of bounding boxes from a dataset in a grid fashion.
    Check if the bounding box is inside the tile's polygon.
    Discard empty patches.
    This should be used with a dataset with only ONE tile.
    """
    def __init__(self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        batch_size: int,
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        ) -> None:

        super().__init__(dataset, size, stride, roi, units)
        self.batch_size = batch_size
        self.dataset = dataset
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        batch = []
        # For each tile...
        for k, hit in enumerate(self.hits): #These hits are all the tiles that intersect the roi (region of interest). If roi not specified then hits = all the tiles

            #print('In sampler')
            #print('tile_polygon: ', tile_polygon)

            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt
            
            discarder_chips = 0
            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    selected_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    #Check if the selected_bbox intersects the tile_polygon (to avoid all black images)
                    selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)

                    #TODO: qui potenzialmente scartare tutte le patch che non hanno edifici o strade
                    #here tile_aoi must be in proj crs
                    if self.dataset.tile_aoi_gdf.intersects(selected_bbox_polygon).any():
                        #print("selected_bbox_polygon", selected_bbox_polygon)
                        batch.append(selected_bbox)
                    else:
                        discarder_chips += 1
                        continue
                    
                    is_last_batch = k == len(self.hits) - 1 and i == rows - 1 and j == cols - 1
                    
                    if len(batch) == self.batch_size or is_last_batch:
                        if is_last_batch and len(batch) < self.batch_size:
                            #print('Last batch not full. Only', len(batch), 'chips')
                            #pad the last batch with the last selected_bbox if it is not full
                            batch.extend([selected_bbox] * (self.batch_size - len(batch)))

                        yield batch
                        batch = []
        print('Discarded empty chips: ', discarder_chips)
        print('True num of batch: ', len(self) - discarder_chips/self.batch_size)

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return math.ceil(self.length / self.batch_size)


# Samplers per Intersection Datasets
class MyIntersectionRandomGeoSampler(RandomGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, size, length, roi, units)
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        i = 0
        while i < len(self):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]

            tile_path1= hit.object[0]
            tile_path2= hit.object[1]

            tile_polyg1 = path_2_tile_aoi(tile_path1)
            tile_polyg2 = path_2_tile_aoi(tile_path2)

            bounds = BoundingBox(*hit.bounds) #TODO: ridurre i bounds usando il bbox del geojson
            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)
            rnd_bbox_polyg = boundingBox_2_Polygon(bounding_box)                
            rnd_central_point = boundingBox_2_centralPoint(bounding_box)

            #se il centro della bounding_box ricade nel polygono del tile1 e in quello del tile2
            # (calcolati usando il geojson) allora la bounding_box è valida
            if rnd_central_point.intersects(tile_polyg1) and rnd_central_point.intersects(tile_polyg2):
                print('In sampler')
                print('tile_polyg1', tile_polyg1)
                print('tile_polyg2', tile_polyg2)
                print()
                i += 1
                yield bounding_box
            
            else:
                continue


class MyIntersectionGridGeoSampler(GridGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        ) -> None:

        super().__init__(dataset, size, stride, roi, units)


    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            path_tile_1 = hit.object[0]
            path_tile_2 = hit.object[1]
            polyg_tile_1 = path_2_tile_aoi(path_tile_1)
            polyg_tile_2=  path_2_tile_aoi(path_tile_2)

            print('In sampler')
            print('tile_polygon 1: ', polyg_tile_1)
            print('tile_polygon 2: ', polyg_tile_2)

            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    selected_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)
                    if selected_bbox_polygon.intersects(polyg_tile_1) and selected_bbox_polygon.intersects(polyg_tile_2):
                        print("selected_bbox_polygon", selected_bbox_polygon)
                        yield selected_bbox
                    else:
                        continue