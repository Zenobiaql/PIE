import json
import os
from datasets import Dataset, DatasetDict, IterableDataset, Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import glob

def VLA_dataset_generator(shards, eos_token, static_video_description, return_info, action_before_vision, wo_text, wo_vision, only_text):

    for shard in shards:
        with open(shard, "r") as f:
            for line in f:

                instance_data = json.loads(line)

                prompt_input = '<|image_1|>\n<|image_2|>\n<bott_i>{}<eott_i>'.format(instance_data['task_description'])
                prompt_output = '<boa_o>{}<eoa_o>'.format(instance_data['answer'])
                image_paths = instance_data['image_paths']

                yield {"prompt": prompt_input, 
                        "answer": prompt_output, 
                        "image_paths": image_paths}

def get_VLA_dataset(args, eos_token, split='train', return_info=False):
    if args.data_root is not None:
        root = args.data_root
        shards = glob.glob(os.path.join(root, split, '*.jsonl'))
    elif args.data_roots is not None:
        shards = []
        for root in args.data_roots:
            shards.extend(glob.glob(os.path.join(root, split, '*.jsonl')))
    else:
        assert False, 'data_root or data_roots must be provided'

    # len_shard = len(shards)
    # shards = shards[:len_shard // 2]
 
    if args.data_debug:
        shards = shards[:1]
    if args.dataset_type == 'dataset':
        ds = Dataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                            "eos_token": eos_token,
                                                            "static_video_description": args.static_video_description,
                                                            "return_info": return_info,
                                                            "action_before_vision": args.action_before_vision,
                                                            "wo_text": args.wo_text,
                                                            "wo_vision": args.wo_vision,
                                                            "only_text": args.only_text
                                                            })
    else: # iterable dataset
        ds = IterableDataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                                "eos_token": eos_token,
                                                                "static_video_description": args.static_video_description,
                                                                "return_info": return_info,
                                                                "action_before_vision": args.action_before_vision,
                                                                "wo_text": args.wo_text,
                                                                "wo_vision": args.wo_vision,
                                                                "only_text": args.only_text
                                                                })
        # ds.column_names = ['text']
    return ds