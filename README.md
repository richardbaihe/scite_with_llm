# scite_with_llm
Distinguish the citation purpose with LLM

## Setup
first, setup openai-api: https://beta.openai.com/docs/api-reference/introduction
pip install openai

## Get Results
`python main.py --api_key <YOUR_KEY>`

## Read Results
There is an example of generation `results/scicite/prompt_json_shot_3_test.jsonl`, whose name means it was evaluated with `json` prompt, and `3` shots.
