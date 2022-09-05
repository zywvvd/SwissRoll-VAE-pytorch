from lib.dataset import DatasetSwissRoll
from lib.generator import MLPGenerator

data = DatasetSwissRoll()

gen = MLPGenerator()
# gen.show()


gt_samples = next(data.batch())
loss = gen.loss(gt_samples)