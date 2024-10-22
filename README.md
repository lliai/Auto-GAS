# Auto-GAS

Auto-GAS: Automated Proxy Discovery for Training-free Generative Architecture Search

This is the code for the paper: Auto-GAS: Automated Proxy Discovery for Training-free Generative Architecture Search.

## Requirements

* Python 3.6
* PyTorch 1.0.0
* CUDA 9.0
* NVIDIA GPU + CUDA CuDNN

## Usage

### Data

Download the dataset.

### Training

Train the search space.

```
python train.py --dataset mnist --data_path data --save_path save
```

### Evaluation

Evaluate the search space.

```
python evaluate.py --dataset mnist --data_path data --save_path save
```


## Acknowledgements

This code is based on the following projects.

* [EAGAN](https://github.com/marsggbo/EAGAN)
* [NASLib](https://github.com/automl/NASLib)
* [TransBench](https://github.com/yawen-d/TransNASBench)

## Citation

If you find Auto-GAS useful in your research, please consider citing the following paper:

```
@inproceedings{li2024auto,
  title={Auto-gas: Automated proxy discovery for training-free generative architecture search},
  author={Li, Lujun and Sun, Haosen and Li, Shiwen and Dong, Peijie and Luo, Wenhan and Xue, Wei and Liu, Qifeng and Guo, Yike},
  year={2024},
  organization={ECCV}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
