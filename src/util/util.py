""" Utility functions """
import os
from shutil import copyfile
import torch
import torch.nn as nn
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, Union


def construct(obj_map: Dict, extra_kwargs: Dict = None) -> object:
    """
    Construct an object of the class specified in 'obj_map' with parameters,
    along with any additional arguments passed in through 'extra_kwargs'.

    Args:
    - obj_map (dict): A dictionary containing the class name under the key 'class'
                      and optionally, arguments under the key 'args'.
    - extra_kwargs (dict, optional): Additional keyword arguments to be passed
                                     to the class constructor.

    Returns:
    - object: An instance of the specified class.
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    classname = obj_map.get("class")
    if classname is None:
        raise ValueError("Class name 'class' not found in obj_map")

    kwargs = obj_map.get("args", {})
    kwargs.update(extra_kwargs)

    class_ = get_from_module(classname)
    if class_ is None:
        raise ValueError(f"Class '{classname}' not found in the module")

    return class_(**kwargs)


def get_from_module(attrname: str) -> Any:
    """
    Return the Python class, method, or attribute of the specified 'attrname'.

    Args:
    - attrname (str): The fully qualified attribute name in the format 'module.ClassName'.

    Returns:
    - Any: The class, method, or attribute corresponding to the given attribute name.

    Raises:
    - AttributeError: If the specified attribute is not found.
    - ImportError: If the specified module is not found.

    Typical usage pattern:
        cls = get_from_module("this.module.MyClass")
        my_class_instance = cls(**kwargs)
    """
    try:
        parts = attrname.split(".")
        module_name = ".".join(parts[:-1])
        module = import_module(module_name)
        return getattr(module, parts[-1])
    except ImportError as e:
        raise ImportError(f"Module {module_name} cannot be found.") from e
    except AttributeError as e:
        raise AttributeError(
            f"Attribute {parts[-1]} cannot be found in {module_name}."
        ) from e


def init_weights(mod):
    """
    Initializes parameters for PyTorch module ``mod``. This should only be
    called when ``mod`` has been newly insantiated has not yet been trained.
    """
    if len(list(mod.modules())) == 0:
        return
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("relu")
            )
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("relu")
            )
            if m.bias is not None:
                m.bias.data.zero_()


def load_state(filename):
    """
    Loads the PyTorch files saved at ``filename``, converting them to be
    able to run with CPU or GPU, depending on the availability on the machine.
    """
    if torch.cuda.is_available():
        return torch.load(filename)
    else:
        return torch.load(filename, map_location=lambda storage, loc: storage)


def save_checkpoint(state_dict, save_dir, filename, is_best):
    """
    Serializes and saves dictionary ``state_dict`` to ``save_dir`` with name
    ``filename``. If parameter ``is_best`` is set to ``True``, then this
    dictionary is also saved under ``save_dir`` as "best.pth".
    """
    save_file = os.path.join(save_dir, filename)
    torch.save(state_dict, save_file)
    if is_best:
        copyfile(save_file, os.path.join(save_dir, "best.pth"))


def try_cuda(x):
    """
    Sends PyTorch tensor or Variable ``x`` to GPU, if available.
    """
    if torch.cuda.is_available():
        return x.cuda()
    return x


def write_vals(outfile, vals, names):
    """
    Writes each value in ``vals[i]`` to a file with name formatted as
    ``outfile.format(names[i])``.
    """

    def write_value(val, outfilename):
        with open(outfilename, "a") as outfile:
            outfile.write("{}\n".format(val))

    for v, n in zip(vals, names):
        write_value(v, outfile.format(n))


def cal_output_size(layers: nn.Sequential, input_size: tuple) -> tuple:
    """
    Calculate the output range of the conv segment

    :param layers: the conv segment of model
    :param input_range: the input range of the conv segment, (channel, height, width)

    :return: the output range of the conv segment, (channel, height, width)
    """
    x = torch.zeros(1, *input_size)
    device = layers[0].weight.device
    x = x.to(device)
    y = layers(x)
    return tuple(y.size()[1:])


def lose_something(
    output_list: List[torch.Tensor],
    lose_index: Optional[Tuple[int]] = None,
    lose_num: Optional[int] = None,
) -> List[torch.Tensor]:
    if lose_index is None or len(lose_index) == 0:
        if lose_num is None:
            return output_list
        else:
            lose_index = torch.randperm(len(output_list))[:lose_num]

    losed_output_list = []

    for i in range(len(output_list)):
        if i in lose_index:
            losed_output_list.append(torch.zeros_like(output_list[i]))
        else:
            losed_output_list.append(output_list[i])
    return losed_output_list
