from base_model.LeNet5 import LeNet5
from util.util import cal_input_shape

# TODO: change the number of split data
split_data_num = 4
# TODO: change the original input shape
input_shape = (1, 28, 28)
class_num = 10

# TODO: change your base model here
model = LeNet5(input_shape, class_num)

# NOTE: the following code is not necessary to change
conv_segment = model.get_conv_segment()
output_shape = model.calculate_conv_output(input_shape)
print('channel, height, width')
print(f"output shape for original input: {output_shape}")
split_output_shape = (
    output_shape[0],
    output_shape[1],
    output_shape[2] // split_data_num,
)
print(f"output shape after split: {split_output_shape}")
split_input_shape = cal_input_shape(conv_segment, split_output_shape)
print(f"input shape after split: {split_input_shape}")
