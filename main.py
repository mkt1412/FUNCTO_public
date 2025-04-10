import os
import yaml
import argparse

from cam_to_target_trans import cam_to_target_trans, demo_test_deteciton
from key_point_track import keyframe_localization
from solve_RT_track import solve_RT
from grasp_point_detection import grasp_point_detection
from func_point_detection import func_point_detect
from func_point_track import func_point_track
from tool_pose_track import tool_center_detect, tool_pose_track
from func_point_transfer import func_point_transfer
from grasp_point_transfer import grasp_point_transfer
from tool_pose_transfer import tool_pose_transfer
from tool_traj_transfer import tool_trajectory_transfer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def demo_processing(config):
    print("############ DEMO PIPELINE ############")

    # Load config
    task_label = config['task_label']
    demo_tool_label = config['demo_tool_label']
    demo_target_label = config['demo_target_label']
    iou_thresh = config['params']['iou_thresh']
    bbox_expansion = config['params']['bbox_expansion']
    num_candidate_keypoints = config['params']['num_candidate_keypoints']
    base_data_path = config['paths']['base_data_path']
    data_path = os.path.join(base_data_path, 'demo_data', f'{task_label}_demo')

    # Step -1: detect demo and test objects
    print("############ Demo and test object detection ############")
    demo_test_deteciton(data_path, demo_tool_label, demo_target_label)

    # Step 0: detect grasping keyframe
    # https://github.com/ddshan/hand_object_detector
    # grasping keyframe index: 16

    # Step 1: transform to target object frame
    print("############ Demo target object frame transformation ############")
    cam_to_target_trans(data_path)

    # Step 2: Demo tool tracking and pre-function keyframe detection
    print("############ Demo tool tracking and pre-function keyframe detection ############")
    keyframe_localization(data_path, demo_tool_label, demo_target_label, iou_thresh, bbox_expansion)

    # Step 3: compute demo tool transformation
    print("############ Demo tool transformation estimation ############")
    solve_RT(data_path)

    # Step 4: grasp point detection
    print("############ Grasp point detection ############")
    grasp_point_detection(data_path, demo_tool_label)

    # Step 5: function point detection
    print("############ Function point detection ############")
    func_point_detect(data_path, demo_tool_label, demo_target_label, task_label, num_candidate_keypoints)

    # Step 6: function point tracking
    print("############ Function point tracking ############")
    func_point_track(data_path)

    # Step 7: tool pose track
    print("############ Tool pose track ############")
    tool_center_detect(data_path)
    tool_pose_track(data_path)

def test_processing(config):
    print("############ TEST PIPELINE ############")

    # Load config
    task_label = config['task_label']
    test_tool_label = config['test_tool_label']
    test_target_label = config['test_target_label']
    num_candidate_keypoints = config['params']['num_candidate_keypoints']
    vp_flag = config['params']['vp_flag']
    sd_dino_flag = config['params']['sd_dino_flag']
    base_data_path = config['paths']['base_data_path']
    data_path = os.path.join(base_data_path, 'demo_data', f'{task_label}_demo')
    test_data_path = os.path.join(base_data_path, 'test_data', f'{task_label}_test')

    # Step 8: transform to target object frame
    print("############ Test target object frame transformation ############")
    demo_test_deteciton(test_data_path, test_tool_label, test_target_label)
    cam_to_target_trans(test_data_path)

    # Step 9: funciton point transfer 
    print("############ Function point transfer ############")
    func_point_transfer(data_path, test_data_path, test_tool_label, test_target_label, task_label, num_candidate_keypoints, sd_dino_flag)

    # Step 10: grasp point transfer
    print("############ Grasp point transfer ############")
    grasp_point_transfer(data_path, test_data_path, test_tool_label, test_target_label, task_label, num_candidate_keypoints, sd_dino_flag)

    # Step 11: tool pose transfer
    print("############ Tool pose transfer ############")
    tool_pose_transfer(data_path, test_data_path, test_tool_label, test_target_label, task_label, vp_flag)

    # Step 12: tool trajectory transfer
    print("############ Tool trajectory transfer ############")
    tool_trajectory_transfer(data_path, test_data_path)

def main():
    parser = argparse.ArgumentParser(description="FUNCTO pipeline")
    parser.add_argument('--task', type=str, default='pour', choices=['pour', 'cut', 'scoop', 'pound', 'brush'],
                        help='Task label: choose from [pour, cut, scoop, pound, brush]')
    parser.add_argument('--stage', type=str, default='test', choices=['demo', 'test', 'all'],
                        help='Which stage to run: demo / test / all')
    args = parser.parse_args()

    # Load config
    config_path = f'./utils_IL/config/config_{args.task}.yaml'
    config = load_config(config_path)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key

    # all/demo/test
    if args.stage == 'demo':
        demo_processing(config)
    elif args.stage == 'test':
        test_processing(config)
    elif args.stage == 'all':
        demo_processing(config)
        test_processing(config)


if __name__ == '__main__':
    main()





    







