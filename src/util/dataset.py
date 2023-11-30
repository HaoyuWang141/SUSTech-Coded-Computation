import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


class PartitionedDataset(Dataset):
    def __init__(
        self,
        images_list: list[torch.Tensor],
        conv_segment_labels: list[torch.Tensor],
        labels: list[any],
    ) -> None:
        self.dataset_num = len(images_list)
        self.images_list = images_list
        self.conv_segment_labels = conv_segment_labels
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor], any]:
        images = [images[idx] for images in self.images_list]
        conv_segment_label = self.conv_segment_labels[idx]
        label = self.labels[idx]

        return images, conv_segment_label, label
