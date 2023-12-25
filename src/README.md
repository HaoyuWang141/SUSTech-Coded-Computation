# Non-linear Erasure-coded Computation for CNN in Distributed Edge Computing

## Quick Start

### 1. Install requirements

```
pip install -r requirements.txt
```
PyTorch is needed. You can install it by following the instructions on the [pytorch website](https://pytorch.org/get-started).

### 2. Choose your TASK (Dataset) and download it

In `./data` folder, there is a `download.py` file. You can simply run it to download all datasets.

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

## Current Problems

1. 需要重新研究数据集的分割方式。不能按空间（长/宽）分割，因为部分模型（VGG16(512, 7, 7), ResNet18(512, 1, 1)）的输出的空间大小无法被整除。

    (1) 可能的解决方式：更早地对卷积层进行分割。例如ResNet18，不让它过池化层就开始分割。目前无法实现。因为在用配置文件训练coder时会直接把分割之前的卷积区的输出张量拉直，这又过不了池化了。
    (2) 把池化层删掉。
    
2. 训练coder时会获取模型的卷积区算分割坐标，此时需要卷积区被解包成一个若干层组成的Sequence。目前的代码直接把Block当成了一个层，需要修改。