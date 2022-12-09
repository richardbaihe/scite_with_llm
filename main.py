
import os
import json
from dataset import SciDataset
from utils import request_completion,wait_for_batch
import argparse
import time
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=3)
    parser.add_argument('--prompt_name', type=str, default='json')
    # parser.add_argument('--test_partition', type=int, default=0)
    parser.add_argument('--api_key', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # experiment settings
    batch_size = 8
    args = get_args()
    k_shot = args.k_shot
    prompt_name = args.prompt_name
    api_key = args.api_key
    test_path = f"./data/scicite/test.jsonl"
    train_path = "./data/scicite/train.jsonl"
    # build dataset
    dataset = SciDataset(train_path, test_path, prompt_name)
    # get results and write into file
    results_file_path = f"./results/scicite/promt_{prompt_name}_shot_{k_shot}_test.jsonl"
    if os.path.exists(results_file_path):
        number_of_line = sum(1 for line in open(results_file_path))
    else:
        number_of_line = 0
    with open(results_file_path, "a") as f:
        time_stamps = []
        for i, example in tqdm(enumerate(dataset)):
            if i < number_of_line:
                continue
            time_stamps.append(time.time())
            wait_for_batch(i, batch_size, time_stamps, batch_time=30)
            rtn = {}
            response = request_completion(api_key, example['string'], 'code-cushman-001', 5, top_p=0.3)
            # time finished the ith example
            time_stamps[-1] = time.time()
            generation = response.choices[0]['text']
            rtn['id'] = example['id']
            rtn['label'] = example['label']
            rtn['generation'] = generation
            f.write(json.dumps(rtn)+'\n')
            time.sleep(1)