import json

from torch.utils.data import Dataset

class SciDataset(Dataset):
    def __init__(self, train_path, test_path, prompt_name='json', k_shot=0):
        self.k_shot = k_shot
        self.prompt_name = prompt_name
        self.raw_data, self.label_space = self.build_dataset(test_path)
        self.k_shot_context = self.get_k_shot_context(train_path)
        self.examples = self.build_examples()

    def input2prompt(self, instance):
        input_string = instance['string']
        try:
            cited_string = input_string[int(instance['citeStart']):int(instance['citeEnd'])]
        except:
            cited_string = 'paper'
        if self.prompt_name == 'json':
            rtn = {}
            rtn['citation purposes'] = self.label_space
            rtn['text'] = input_string
            # cited_string = input_string[int(instance['citeStart']):int(instance['citeEnd'])]
            # rtn['question'] = f"What is the citation purpose of {cited_string} in text?"
            rtn['question'] = "What is the citation purpose of the text above?"
            rtn['answer'] = instance['label']
            rtn = json.dumps(rtn, sort_keys=False)
        
            part_b = instance['label']+'"}'
            part_a = rtn[:-len(part_b)]
        elif self.prompt_name == 'json_v2':
            rtn = {}
            rtn['choices'] = self.label_space
            rtn['text'] = input_string.replace('\n','')
            # cited_string = input_string[int(instance['citeStart']):int(instance['citeEnd'])]
            rtn['question'] = f"What is the citation purpose of {cited_string} in text?"
            rtn['answer'] = instance['label']
            rtn = json.dumps(rtn, sort_keys=False)
        
            part_b = instance['label']+'"}'
            part_a = rtn[:-len(part_b)]
        elif self.prompt_name == 'original_json':
            rtn = {}
            rtn['string'] = instance['string']
            rtn['label'] = instance['label']
            rtn = json.dumps(rtn, sort_keys=False)
            part_b = instance['label']+'"}'
            part_a = rtn[:-len(part_b)]
        elif self.prompt_name == 'multi_choice':
            part_a = f"Text: {input_string}\nQ: What is the citation purpose of the text above? background, result, or method?\nA: "
            part_b = instance['label']
        elif self.prompt_name == 'multi_choice_single_line':
            part_a = f"Text: {input_string} Q: what is the citation purpose of the text above? Background, result, or method? A: "
            part_b = instance['label']
        else:
            raise NotImplementedError(f"prompt_name={self.prompt_name} is not implemented")
        # part_a is the prompt, part_b is the answer
        return part_a, part_b


    def input2prompt_logprobs(self, instance):
        input_string = instance['string']
        try:
            cited_string = input_string[int(instance['citeStart']):int(instance['citeEnd'])]
        except:
            cited_string = 'paper'
        part_a_list = []
        part_b_list = []
        if 'logprobs' not in self.prompt_name:
            raise ValueError(f"prompt_name={self.prompt_name} should contain logprobs")
        if self.prompt_name == 'cloze_style_logprobs':
            for label in self.label_space:
                part_a = f"The citation purpose of '{cited_string}' in the following text is {label}: "
                part_b = f"{input_string}".replace('\n', ' ')
                part_a_list.append(part_a)
                part_b_list.append(part_b)
        elif self.promt_name == 'mcq_style_logprobs':
            for label in self.label_space:
                part_a = f"Text: {input_string} Q: what is the citation purpose of the text above? Background, result, or method? A: "
                part_b = instance['label']
                part_a_list.append(part_a)
                part_b_list.append(part_b)
        else:
            raise NotImplementedError(f"prompt_name={self.prompt_name} is not implemented")
        return part_a_list, part_b_list

    def build_examples(self):
        examples = []
        for instance in self.raw_data:
            example = {'id': instance['id'], 'label': instance['label']}
            if 'logprobs' in self.prompt_name:
                part_a_list, part_b_list = self.input2prompt_logprobs(instance)
                input_string_list = [part_a+part_b for part_a, part_b in zip(part_a_list, part_b_list)]
                if self.k_shot>0:
                    example['string'] = [self.k_shot_context+'\n'+input_string for input_string in input_string_list]
                    example['logprobs_start_index'] = [len(self.k_shot_context)+1+len(part_a) for part_a in part_a_list]
                    example['logprobs_end_index'] = [len(self.k_shot_context)+1+len(part_a)+len(part_b) for part_a, part_b in zip(part_a_list, part_b_list)]
                else:
                    example['string'] = input_string_list
                    example['logprobs_start_index'] = [len(part_a) for part_a in part_a_list]
                    example['logprobs_end_index'] = [len(part_a)+len(part_b) for part_a, part_b in zip(part_a_list, part_b_list)]
            else:
                part_a, part_b = self.input2prompt(instance)
                if self.k_shot>0:
                    example['string'] = self.k_shot_context+'\n'+part_a
                else:
                    example['string'] = part_a
            examples.append(example)
        return examples

    def get_k_shot_context(self, data_path):
        if self.k_shot>0:
            assert self.k_shot % len(self.label_space) == 0, f"k_shot={self.k_shot} should be a multiple of the number of classes={self.label_space}"
            k_shot_labels = []
            for i in range(1,self.k_shot//len(self.label_space)+1):
                k_shot_labels+=self.label_space
            k_shot_context = []
            with open(data_path, "rb") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    if data['label'] == k_shot_labels[0]:
                        if 'logprobs' in self.prompt_name:
                            part_a_list, part_b_list = self.input2prompt_logprobs(data)
                            input_string_list = [part_a+part_b for part_a, part_b in zip(part_a_list, part_b_list)]
                            part_a = input_string_list[self.label_space.index(data['label'])]
                            part_b = ''
                        else:
                            part_a, part_b = self.input2prompt(data)
                        k_shot_context.append(part_a+part_b)
                        k_shot_labels = k_shot_labels[1:]
                    else:
                        continue
                    if len(k_shot_labels) == 0:
                        break
            return '\n'.join(k_shot_context)
        else:
            return ''
        
    def build_dataset(self, file_path):
        # loading data
        label_space = []
        data = []
        with open(file_path, "rb") as f:
            for line in f.readlines():
                data.append(json.loads(line))
                if data[-1]['label'] not in label_space:
                    label_space.append(data[-1]['label'])
        return data, label_space

    def __getitem__(self, idx):
        return self.examples[idx]
    def __len__(self):
        return len(self.examples)