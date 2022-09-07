###  Implementation of VAE on Swiss Roll dataset 

#### dataset

```python
from sklearn.datasets import make_swiss_roll
swiss_roll_samples, _ = make_swiss_roll(size, noise=0.3)
```

- data looks like in 2D

![](https://101.43.39.125/HexoFiles/vvd_file_mt/202209071610555.jpg)

### requirement

- torch 1.8+

#### usage

```
python train
```

#### results

![](https://101.43.39.125/HexoFiles/vvd_pc_upload/vae-res.gif)

![](https://101.43.39.125/HexoFiles/vvd_file_mt/202209071827475.jpg)

### References

- https://github.com/AntixK/PyTorch-VAE