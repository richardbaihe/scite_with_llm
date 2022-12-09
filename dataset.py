import json

from torch.utils.data import Dataset

class SciDataset(Dataset):
    def __init__(self, train_path, test_path, prompt_name='json', k_shot=0):
        self.k_shot = k_shot
        self.prompt_name = prompt_name
        self.raw_data, self.label_space = self.build_dataset(test_path)
        self.k_shot_context = self.get_k_shot_context(train_path)
        self.examples = self.build_examples()
    
    def get_k_shot_context(self, data_path):
        if self.k_shot>0:
            assert k_shot // len(label_space) == 0, "k_shot should be a multiple of the number of classes"
            k_shot_labels = label_space
            for i in range(1,k_shot+1):
                k_shot_labels+=label_space
            k_shot_context = []
            with open(data_path, "rb") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    if data['label'] == k_shot_labels[0]:
                        k_shot_context.append(self.input2prompt(data))
                        k_shot_labels = k_shot_labels[1:]
                    else:
                        continue
                    if len(k_shot_labels) == 0:
                        break
            return '\n'.join(k_shot_context)
        else:
            return ''
        
    def build_examples(self):
        examples = []
        for instance in self.raw_data:
            example = {'id': instance['id'], 'label': instance['label']}
            prompted_instance = self.input2prompt(instance)
            if self.prompt_name == 'json':
                prompted_instance = prompted_instance[:-2]
            if self.k_shot>0:
                example['string'] = self.k_shot_context+'\n'+prompted_instance
            else:
                example['string'] = prompted_instance
            examples.append(example)
        return examples
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

    def input2prompt(self, instance):
        input_string = instance['string']

        rtn = {}
        rtn['citation purposes'] = self.label_space
        rtn['text'] = input_string
        # cited_string = input_string[int(instance['citeStart']):int(instance['citeEnd'])]
        # rtn['question'] = f"What is the citation purpose of {cited_string} in the text above?"
        rtn['question'] = "What is the citation purpose of the text above?"
        rtn['answer'] = ''
        rtn = json.dumps(rtn, sort_keys=False)
        return rtn
    def __getitem__(self, idx):
        return self.examples[idx]
    def __len__(self):
        return len(self.examples)