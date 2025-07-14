import json
import os
import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from torch_geometric.io import read_txt_array

import torch_geometric.transforms as T

import glob
import re
from pathlib import Path
import torch
import numpy as np
import cv2

class VoronoiNet(InMemoryDataset):

    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['processed_data.pt']
    
    def process(self) -> None:

        def load_data_from_folders():
            base_path = "Vf50_R4"
            data_collection = []
            
            # Generate folder names from 1 to 1000
            for i in range(1, 1001):
                folder_name = f"0.5_4.0_{i}"
                folder_path = Path(base_path) / folder_name
                
                if not folder_path.exists():
                    continue
                    
                # Load both files from the folder
                try:
                    geometry_path = folder_path / "geometry.th"
                    max_f_path = folder_path / "max_f.th"
                    
                    if geometry_path.exists() and max_f_path.exists():
                        geometry_data = torch.load(geometry_path)
                        max_f_data = torch.load(max_f_path)
                        
                        data_collection.append({
                            'folder': folder_name,
                            'geometry': geometry_data,
                            'max_f': max_f_data
                        })
                except Exception as e:
                    print(f"Error loading data from {folder_name}: {e}")
            
            return data_collection
        
        def find_centroids(image):
            image = image.to('cpu').numpy().astype(np.uint8)
            binary_image = (image > 0.5).astype(np.uint8)
            
            # Connected component analysis on the inverted image to find dark circles
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(1 - binary_image, connectivity=8)

            # Convert centroids to list of tuples, skipping background (index 0)
            valid_centroids = [(x/512, y/512) for x, y in centroids[1:]]
            
            return valid_centroids
      
        data_list = []

        data_list = load_data_from_folders()

        dataset = []

        # Calculate mean and standard deviation of max_f values across the dataset
        max_f_values = torch.tensor([data['max_f'] for data in data_list])
        f_max = torch.max(max_f_values)
        f_min = torch.min(max_f_values)
        f_range = f_max - f_min

        for data in data_list:
            data['max_f'] = (data['max_f'] - f_min) / f_range

        # # Normalize max_f values in data_list
        # for data in data_list:
        #     data['max_f'] = (data['max_f'] - mean_max_f) / std_max_f

        # print(f"mean_max_f: {mean_max_f}, std_max_f: {std_max_f}")
        print(f"f_max: {f_max}, f_min: {f_min}, f_range: {f_range}")

        for data in data_list:
            pos = torch.tensor(find_centroids(data['geometry']),dtype=torch.float32)
            x = pos
            y = data['max_f'].clone().detach().to(torch.float32)
            data = Data(pos=pos, x=x, y=y)
            pre_transform = T.KNNGraph(k=5)
            # pre_transform = T.RadiusGraph(r=0.5, loop=False)
            data = pre_transform(data)
            distance = T.Distance(norm=False)
            local_cartesian = T.LocalCartesian(norm=False)
            data = distance(data)
            data = local_cartesian(data)
            data.edge_attr[:,0] = 1/data.edge_attr[:,0]
            data.edge_index = data.edge_index.to(torch.int64)
            dataset.append(data)

        self.save(dataset,'processed/processed_data.pt')


#    def len(self):
#        return len(self.dataset)
#
#    def get(self, idx):
#        return self.dataset[idx]
