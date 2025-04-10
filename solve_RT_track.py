import numpy as np
import json
import os

from utils_IL.geometry_utils import ransac_rigid_transform_3D

def solve_RT(data_path):

    output_path = os.path.join(data_path, 'solve_rt_output')
    os.makedirs(output_path, exist_ok=True)
    
    tracked_points_path = os.path.join(data_path, 'key_point_track_output', 'tracked_visible_points.json')
    with open(tracked_points_path, 'r') as json_file:
        tracked_points = json.load(json_file)
    
    trans_data = {}
    for frame_idx in tracked_points.keys():
        if frame_idx == '0':
            continue  # first frame as the reference
            
        print(f"processing frame {frame_idx}......")
            
        cur_idx = str(int(frame_idx))

        trans, inliers = ransac_rigid_transform_3D(np.array(tracked_points['0']).T, np.array(tracked_points[cur_idx]).T)

        trans_data[cur_idx] = trans.tolist()

    with open(os.path.join(output_path, 'frame_transformations.json'), 'w') as json_file:
        json.dump(trans_data, json_file, indent=4)
    
    return None
    





    

