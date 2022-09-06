from sklearn.datasets import make_swiss_roll
from .utils import show


class DatasetBase(object):
    def gen_data_xy(self, size=512):
        raise NotImplementedError('you need implement gen_data_xy')

    def __len__(self):
        return len(self.samples)

    def show(self):
        samples = self.gen_data_xy()
        show(samples, type(self).__name__)


class DatasetSwissRoll(DatasetBase):
    def gen_data_xy(self, size=1024):
        swiss_roll_samples, _ = make_swiss_roll(size, noise=0.3)
        samples = swiss_roll_samples[:,[0,2]]
        return samples
