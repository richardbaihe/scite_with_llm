import openai
import time
import numpy as np

def wait_for_batch(data_index,batch_size,time_stamp_list, batch_time=60):
    if len(time_stamp_list) > batch_size:
        current_time = time.time()
        while current_time-time_stamp_list[0] < batch_time:
            time.sleep(batch_time-(current_time-time_stamp_list[0]))
            current_time = time.time()
        time_stamp_list.pop(0)
        time_stamp_list[-1] = current_time
        

def request_completion(api_key, prompt, engine, max_tokens, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=1,
        echo=True
    )
    return response

def get_generation_from_response_logprob(response, label_space, start_index, end_index):
    log_probs_list = []
    
    for choice,s,e in zip(response['choices'],start_index,end_index):
        log_probs = 0
        for token_logp, text_offset in zip(choice['logprobs']['token_logprobs'],choice['logprobs']['text_offset']):
            if s <= text_offset < e:
                log_probs += token_logp
        log_probs_list.append(log_probs)
    log_probs_list = np.array(log_probs_list)
    label = label_space[np.argmax(log_probs_list)]
    return label