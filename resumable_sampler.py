import torch
import numpy as np

class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    # data_source: Sized
    # replacement: bool

    def __init__(self, data_source, replacement=True):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(42)
        self.replacement = replacement

        self.perm_index = 0
        # self.perm = torch.randperm(self.num_samples, generator=self.generator)
        # self.perm = torch.from_numpy(np.random.choice(self.data_source, self.num_samples, p=self.data_source.weights))
        self.perm = torch.FloatTensor(self.data_source.weights).multinomial(num_samples=self.num_samples, replacement=self.replacement)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            # self.perm = torch.FloatTensor(self.data_source.weights).multinomial(num_samples=self.num_samples, replacement=self.replacement)

            # self.perm = torch.from_numpy(np.random.choice(self.data_source, self.num_samples, p=self.data_source.weights))

            # self.perm = torch.randperm(self.num_samples, generator=self.generator)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}

    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])
