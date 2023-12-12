# Non-linear Erasure-coded Computation for CNN in Distributed Edge Computing

## Quick Start

### 1. Install requirements

```
pip install -r requirements.txt
```
PyTorch is needed. You can install it by following the instructions on the [pytorch website](https://pytorch.org/get-started).

### 2. Choose your TASK (Dataset) and download it

In `./data` folder, there is `download.py` file. You can download your dataset.

### 3. Choose a base model and train it

In `./base_model` folder, there are some base models. You can choose one of them or implement your own base model. Then you need to train your base model and save it in `./base_model/<MODEL>/<TASK>[/<DATE>]` folder. You can refer the `./test/train_base_model.ipynb` script to train your base model.

### 4. Split your dataset

In `./data/MNIST` folder, you can find the `split` directory. Split dataset is in this directory. You need to split the original dataset by yourself. You can refer `./data/split_data.py` to split data. The output path (`output_file=` in method `split_data`) is `./data/<DATASET>/split/<SPLIT NUM>/split_<test/train>_datasets.pt` folder.

### 5. Choose encoder and decoder

### 6. Write your config file

In `./config` folder, there are `README.md` and `eg.config.json` file. You can write your own config file according to the examples. The details of the config file are described in `README.md`.

### 7. calculate the split input shape

You need to calculate the split input shape by hand and write it into your config file (`general -> split_data_shape`). You can use the `./cal_split_data_shape.py` script to calculate the split input shape. You need to overwrite something in this script marked by `TODO`.

### 8. Run main.py

```
python ./main.py <config path>

e.g.
python ./main.py config/minist/base_model-LeNet5/eg.config.json
```

