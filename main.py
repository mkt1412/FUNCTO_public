import os
import yaml
import argparse

from cam_to_target_trans import cam_to_target_trans, demo_test_deteciton
from func_point_transfer import func_point_transfer
from grasp_point_transfer import grasp_point_transfer
from tool_pose_transfer import tool_pose_transfer
from tool_traj_transfer import tool_trajectory_transfer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def test_processing(config):

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
    parser.add_argument('--task', type=str, default='pour')
    args = parser.parse_args()

    # Load config
    config_path = f'./utils_IL/config/config_{args.task}.yaml'
    config = load_config(config_path)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key

    test_processing(config)


if __name__ == '__main__':
    main()





    







