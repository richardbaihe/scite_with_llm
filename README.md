# scite_with_llm
Distinguish the citation purpose with LLM

## Setup
First, setup openai-api and get your key: https://beta.openai.com/docs/api-reference/introduction

`pip install openai`

## Get Results
`python main.py --api_key <YOUR_KEY> --k_shot 3 --prompt_name json --engine code-cushman-001` 

## Read Results
There is an example of generation `results/scicite/prompt_json_shot_3_test.jsonl`, whose name means it was evaluated with `json` prompt, and `3` shots.

## Evaluate Results
Install package  
`pip install scikit-learn`  
then run  
`python eval.py`   
will evaluate all the result files under `./results/scicite`
