# Configuration File Documentation

This document outlines the structure of the configuration file and the purpose of each section. The configuration file is in JSON format and includes the following main sections: `WORK_FLOW`, `general`, `train`, `test`, `save`, `load`, and `base`. `WORK_FLOW` and `general` are necessary while others are optional.

## WORK_FLOW
`WORK_FLOW` defines the order of operations in the workflow. It is an array of strings, which includes:
- `base`: Test basic model and get the base line of accuracy.

- `train`: Train encoder and decoder.

- `test`: Test encoder and decoder.

- `save`: Save training state, which contains:
  - `epoch`: the current epoch
  
  - `encoder_state_dict`
  
  - `decoder_state_dict`
  
  - `encoder_optimizer_state_dict`
  
  - `decoder_optimizer_state_dict`
  
  - `loss_history`: 
  
    [

​			[epoch1_loss1, epoch1_loss2, ...], 

​			[epoch2_loss1, epoch2_loss2, ...], 

​			...

​		]

- `load`: Load training state from file

## general
The `general` section includes general configuration settings, which are all necessary:
- `data_split_num`: The number of data splits, an integer.
- `redundancy_num`: The redundancy number, an integer.
- `batch_size`: Batch size for processing, an integer.
- `split_data_shape`: The shape of the split data, an array of integers.
- `base_model`: The Base model, specified by `class` along with `args`, and then it will load `state dict` from `path`.
- `encoder`: Encoder, specified by `class`.
- `decoder`: Decoder, specified by `class`.

## train
The `train` section includes settings for the training process:
- `train_dataset`: Configuration for the training dataset.
- `encoder_optimizer` and `decoder_optimizer`: Settings for the optimizer, including its `class` and `args` such as learning rate (`lr`) and `momentum`.
- `criterion`: The loss function, specified by its `class`.
- `scheduler`: Learning rate scheduler settings.
- `epoch_num`: The number of training epochs.
- `save_interval`: Interval for saving the model (not implement now).

## test
The `test` section includes settings for testing:
- `test_dataset`: Configuration for the testing dataset.
- `lose_device_index`: The index of the devices to lose, an array.
- `lose_device_num`: The number of devices to lose, will be random choice later, an integer.

`lose_device_num` only works when `lose_device_index` is `null`.

If `lose_device_num`  and `lose_device_index` are both `null`, it means no data from distributed devices is lost.

## save
The `save` section includes settings for saving the model:
- `save_dir`: Directory to save the model.

## load
The `load` section includes settings for loading the model:
- `load_path`: Path from which to load the model.

## base
The `base` section includes the testing for base model. It used to get the base line of Accuracy.

+ `dataset`: the dataset for testing base model. It also needs to be `dataset.splited_dataset.SplitedTestDataset`, but `split_num` is 1.