from torch.utils.data import DataLoader

from data.dataset import MIDILoopDataset


class CustomLoader(DataLoader):
    def __init__(self, dataset, params):
        self.dataset = dataset
        if params.overfit:
            self.dataset = MIDILoopDataset(
                [dataset[params.overfit_index]] * params.overfit_length
            )

        super().__init__(
            self.dataset,
            params.batch_size,
            params.shuffle,
            num_workers=params.num_workers,
        )
