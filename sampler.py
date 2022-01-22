import torch
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


class MultitaskSampler(Sampler):
    r"""Samples elements for multi-task learning.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool) or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, "
                "but got drop_last={}".format(drop_last)
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch_dict = {}
        for idx in torch.randperm(len(self.data_source)).tolist():
            task = self.data_source[idx]["task"]
            if task not in batch_dict:
                batch_dict[task] = []
            batch_dict[task].append(idx)
            for batch in batch_dict.values():
                if len(batch) == self.batch_size:
                    yield batch
                    batch_dict[task] = []
        for batch in batch_dict.values():
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            num_samples_dict = {}
            for data in self.data_source:
                task = data["task"]
                if task not in num_samples_dict:
                    num_samples_dict[task] = 0
                num_samples_dict[task] += 1
            return sum(
                num_samples // self.batch_size for num_samples
                in num_samples_dict.values()
            )
        else:
            return (
                (len(self.data_source) + self.batch_size - 1)
                // self.batch_size
            )


class WeightedMultitaskSampler(Sampler):
    r"""Samples elements for multi-task learning.

    Args:
        tasks (sequence): a sequence of tasks to learn
        weights (sequence): a sequence of weights, not necessary summing up to one
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, tasks, weights, data_source, batch_size, drop_last):
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool) or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, "
                "but got drop_last={}".format(drop_last)
            )
        self.tasks = tasks
        self.weights = torch.as_tensor(weights, dtype=torch.double)

        if len(self.tasks) != len(self.weights):
            raise ValueError(
                "the length of weights should be the same as that of tasks"
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        # calculate num_samples
        nums_samples = [0] * len(self.tasks)
        for data in self.data_source:
            if data["task"] not in self.tasks:
                raise ValueError(
                    "There is an invalid task that tasks does not contain "
                    "in the dataset"
                )
            nums_samples[self.tasks.index(data["task"])] += 1
        
        n = len(self.tasks)
        s = torch.as_tensor(nums_samples, dtype=torch.double)
        w = self.weights
        S = s.repeat(n, 1) / s.view(n, -1)
        W = w.repeat(n, 1) / w.view(n, -1)
        i = (S - W >= 0).prod(1).nonzero().item()
        self.maxs_samples = (s[i] * W[i, :]).type(torch.int).tolist()

    def __iter__(self):
        batches = [[] for ti in range(len(self.tasks))]
        sums_samples = [0] * len(self.tasks)
        for idx in torch.randperm(len(self.data_source)).tolist():
            task_idx = self.tasks.index(self.data_source[idx]["task"])
            if sums_samples[task_idx] < self.maxs_samples[task_idx]:
                batches[task_idx].append(idx)
                sums_samples[task_idx] += 1
            for ti in range(len(self.tasks)):
                if (
                    sums_samples[ti] == self.maxs_samples[ti]
                    and len(batches[ti]) > 0 and not self.drop_last
                    or len(batches[ti]) == self.batch_size
                ):
                    yield batches[ti]
                    batches[ti] = []
        for batch in batches:
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
            sum_samples = 0
            for ti in range(len(self.tasks)):
                if self.drop_last:
                    sum_sample = self.maxs_samples[ti] // self.batch_size
                else:
                    sum_sample = (
                        (len(self.maxs_samples[ti]) + self.batch_size - 1)
                        // self.batch_size
                    )
                sum_samples += sum_sample
            return sum_samples
