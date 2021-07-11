import os

import torch
from torch.utils.data import Dataset

data_dict = {
    'cfemotion': 'cf12_half',
    'emotion': 'tec',
    'response': 'dd',
    'sentiment': 'sst2_qtr'
}


class MultitaskDataset(Dataset):
    def __init__(self, tasks, tokenizer, data_dir, type_path, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.labels = []
        self.tasks = []

        self._build(tasks, data_dir, type_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        task = self.tasks[index]

        source_ids = self.inputs[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze

        target_ids = self.targets[index]['input_ids'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()  # might need to squeeze

        target_label = self.labels[index]

        return {
            'task': task,
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'target_label': target_label
        }

    def _build(self, tasks, data_dir, type_path):
        for task in tasks:
            input_path = os.path.join(
                data_dir,
                data_dict[task],
                type_path + '.source'
            )

            with open(input_path) as f:
                for line in f:
                    input_ = line.strip()

                    # tokenize inputs
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                        [input_],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True
                    )

                    self.inputs.append(tokenized_inputs)

            target_path = os.path.join(
                data_dir,
                data_dict[task],
                type_path + '.target'
            )

            with open(target_path) as f:
                for line in f:
                    target = ' '
                    label = -1

                    if task == 'response':
                        target = line.rstrip()
                    else:
                        label = int(line)

                    # tokenize targets
                    tokenized_targets = self.tokenizer.batch_encode_plus(
                        [target],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True
                    )
                    self.targets.append(tokenized_targets)
                    self.labels.append(torch.tensor([label]))

                    self.tasks.append(task)
