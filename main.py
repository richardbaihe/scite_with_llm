import os
import json
from dataset import SciDataset
from utils import request_completion, wait_for_batch, get_generation_from_response_logprob
import argparse
import time
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=3)
    # prompt_name: json, multi_choice, original_json
    parser.add_argument('--prompt_name', type=str, default='json')
    # parser.add_argument('--test_partition', type=int, default=0)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--engine', type=str, default='code-cushman-001')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # experiment settings    
    batch_size = 8
    args = get_args()
    k_shot = args.k_shot
    prompt_name = args.prompt_name
    api_key = args.api_key
    engine = args.engine
    test_path = f"./data/scicite/test.jsonl"
    train_path = "./data/scicite/train.jsonl"
    # build dataset
    dataset = SciDataset(train_path, test_path, prompt_name, k_shot)
    # get results and write into file
    results_file_path = f"./results/scicite/engine_{engine}_prompt_{prompt_name}_shot_{k_shot}_test.jsonl"
    if os.path.exists(results_file_path):
        number_of_line = sum(1 for line in open(results_file_path))
    else:
        number_of_line = 0

    with open(results_file_path, "a") as f:
        time_stamps = []
        for i, example in tqdm(enumerate(dataset), total=len(dataset)):
            # skill the line that already requested
            if i < number_of_line:
                continue
            time_stamps.append(time.time())
            # process #batch_size examples every #batch_time seconds
            wait_for_batch(i, batch_size, time_stamps, batch_time=30)
            rtn = {}

            if type(example['string']) == list:
                input_string = example['string'][0]
            else:
                input_string = example['string']
            length = len(input_string)
            # print an example in log
            if i == 0:
                print(example['string'])
            try:
                response = request_completion(api_key, example['string'], engine, 5, top_p=0.3)
            except:
                tqdm.write('openai.error.RateLimitError, wait for 59 seconds')
                time.sleep(59)
                response = request_completion(api_key, example['string'], engine, 5, top_p=0.3)

            # time finished the ith example
            time_stamps[-1] = time.time()
            if 'logprob' in prompt_name:
                label_string = get_generation_from_response_logprob(response, dataset.label_space, example['logprobs_start_index'], example['logprobs_end_index'])
                generation = input_string+label_string
            else:
                generation = response.choices[0]['text']
            rtn['id'] = example['id']
            rtn['label'] = example['label']
            rtn['generation'] = generation
            rtn['input_length'] = length
            f.write(json.dumps(rtn) + '\n')
            time.sleep(1)