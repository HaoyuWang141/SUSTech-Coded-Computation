import torch
from torch.utils.data import Dataset


class SplitedDataset(Dataset):
    def __init__(
        self,
        images_list: list[torch.Tensor] = None,
        labels: list[any] = None,
    ) -> None:
        self.images_list = images_list
        self.labels = labels

        if images_list is None:
            self.has_init = False
            self.split_num = None
            self.data_num = None
            self.data_shape = None
        else:
            self.has_init = True
            self.split_num = len(images_list)
            self.data_num = len(images_list[0])
            self.data_shape = tuple(images_list[0][0].size())

    def __len__(self) -> int:
        if self.has_init is False:
            raise Exception("Dataset is not initialized.")

        return self.data_num

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor], any]:
        if self.has_init is False:
            raise Exception("Dataset is not initialized.")

        image_list = [images[idx] for images in self.images_list]
        label = self.labels[idx] if self.labels is not None else None

        return image_list, label

    def describe(self) -> str:
        if self.has_init is False:
            raise Exception("Dataset is not initialized.")

        return f"Dataset: split_num={self.split_num}, data_num={self.data_num}, data_shape={self.data_shape}"

    def save(self, path: str) -> None:
        if self.has_init is False:
            raise Exception("Dataset is not initialized.")

        torch.save(
            {
                "images_list": self.images_list,
                "labels": self.labels,
                "split_num": self.split_num,
                "data_num": self.data_num,
                "data_shape": self.data_shape,
            },
            path,
        )

    def load(self, path: str, device: torch.device = None) -> object:
        checkpoint = torch.load(path, map_location=device)
        self.images_list = checkpoint["images_list"]
        self.labels = checkpoint["labels"]
        self.split_num = checkpoint["split_num"]
        self.data_num = checkpoint["data_num"]
        self.data_shape = checkpoint["data_shape"]

        self.has_init = True
        
        return self


class SplitedTestDataset(SplitedDataset):
    def describe(self) -> str:
        return (
            f"Splited Test Dataset: "
            + f"split_num={self.split_num}, data_num={self.data_num}, data_shape={self.data_shape}"
        )


class SplitedTrainDataset(SplitedDataset):
    def describe(self) -> str:
        return (
            f"Splited Train Dataset: "
            + f"split_num={self.split_num}, data_num={self.data_num}, data_shape={self.data_shape}"
        )
