from lib.dataset import DatasetSwissRoll
from lib.generator import MLPGenerator

data = DatasetSwissRoll()

gen = MLPGenerator()
# gen.show()


gt_samples = next(data.batch(size=32))
loss = gen.loss(gt_samples)
print(loss)