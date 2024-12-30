import os
import argparse
import json
import random

def reset_gripper_width(x):
        return '0' if x > 0.07 else '1'
    
def format_action(values, grip):
    formatted_str = '['

    for value in values:
        # 四舍五入并乘以10000
        rounded_value = round(value, 4)
        int_value = int(rounded_value * 10000)
        
        # 格式化
        if int_value >= 0:
            formatted_value = f"+{int_value:03d}"
        else:
            formatted_value = f"{int_value:04d}"
        formatted_str += formatted_value + ','

    formatted_str += grip
    formatted_str += ']'
    
    return formatted_str

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='/mnt/data-rundong/robot_datasets/tokenizer-training/pizza_width_split/')
parser.add_argument('--dst', type=str, default='/mnt/data-rundong/robot_datasets/tokenizer-training/pizza_preprocessed_for_pie/')

args = parser.parse_args()

for split in ["train", "test"]:
    src_path = args.src
    for i in range(2):
        j = "train" if i == 0 else "test"
        src_filepath = os.path.join(src_path, j, f'{j}.jsonl')
        f = open(src_filepath, 'r')
        dst_filepath = os.path.join(args.dst, j,  f'{j}.jsonl')
        os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        dst_file = open(dst_filepath, 'w')

        # read and store all lines
        for line in f:
            instance_data = json.loads(line)

            image_root = '/mnt/robotdata/datasets/pizza_robot'
            # prompt_input = '<|image_1|>\n<|image_2|>\n<bott_i>' + instance_data['task_description'] + '<eott_i>'
            task_description = instance_data['task_description']
            # prompt_output_format = '<boa_o>{}<eoa_o>'
            prompt_output_format = '{}'
            image_format = image_root + '/' + str(instance_data['ID']) + '/' + str(instance_data['trajectory_id']) + '/images/right_rgb' + '/{:03d}' + '.jpg'

            num_frames = instance_data["frame_number"]
            num_input_interval = 3
            num_pred_actions = 6

            # 去掉最后补全用的重复帧
            prev_frame_id = -100
            for frame_pos in range(num_frames):
                cur_frame_id = instance_data['image_indices'][frame_pos]
                if cur_frame_id == prev_frame_id: # 重复
                    num_frames = frame_pos
                    break
                # 未重复
                prev_frame_id = cur_frame_id

            num_start = num_frames ###########
            for start in range(-1, num_start):
                images = []
                prompt_output_action = ''
                # try:
                if start == -1:
                    img_start = image_format.format(instance_data['image_indices'][0])
                    if not os.path.exists(img_start):
                        continue
                    images = [img_start] * 2
                    
                    pred_action_start_idx = 0 # 预测的action开始的index，注意是image_indices中的顺序而不是实际的frame_id
                    pred_action_end_idx = pred_action_start_idx + num_pred_actions - 1
                    if pred_action_end_idx >= num_start:
                        continue # 不到一个clip的数据，太短没有意义

                    pred_action_text = ''

                    for pred_action_idx in range(pred_action_start_idx, pred_action_end_idx + 1):
                        pred_xyzrpy_vec = instance_data['actions'][pred_action_idx][:-1]
                        pred_gripper = reset_gripper_width(instance_data['action_gripper'][pred_action_idx][-1])
                        pred_action_text += format_action(pred_xyzrpy_vec, pred_gripper) # e.g. [+000,-005,-001,+050,+002,+007,+788,0]
                        pred_action_text += ','
                    
                    prompt_output_action = prompt_output_format.format(pred_action_text)

                else:
                    img_start_idx = start
                    img_end_idx = img_start_idx + num_input_interval
                    if img_end_idx >= num_start:
                        continue
                    img_start = image_format.format(instance_data['image_indices'][img_start_idx])
                    img_end = image_format.format(instance_data['image_indices'][img_end_idx])
                    if not os.path.exists(img_start):
                        continue
                    if not os.path.exists(img_end):
                        continue
                    images = [img_start, img_end]

                    pred_action_start_idx = img_end_idx 
                    pred_action_end_idx = pred_action_start_idx + num_pred_actions - 1

                    pred_action_text = ''

                    for pred_action_idx in range(pred_action_start_idx, pred_action_end_idx + 1):
                        if pred_action_idx >= num_start: # 超出边界
                            pred_xyzrpy_vec = [0. for _ in range(6)]
                            pred_gripper = '0' # 默认静止，夹爪闭合
                        else:
                            pred_xyzrpy_vec = instance_data['actions'][pred_action_idx][:-1]
                            pred_gripper = reset_gripper_width(instance_data['action_gripper'][pred_action_idx][-1])

                        pred_action_text += format_action(pred_xyzrpy_vec, pred_gripper) # e.g. [+000,-005,-001,+050,+002,+007,0]
                        pred_action_text += ','
                        
                    prompt_output_action = prompt_output_format.format(pred_action_text)

                stacked_instance = {}
                stacked_instance["task_description"] = task_description
                stacked_instance["answer"] = prompt_output_action
                stacked_instance["image_paths"] = images
                dst_file.write(json.dumps(stacked_instance) + '\n')