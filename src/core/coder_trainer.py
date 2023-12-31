from util.util import construct, lose_something
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import datetime
import os
from dataset.image_dataset import ImageDataset
import torchvision


class CoderTrainer:
    def __init__(self, config):
        self.config = config

        self.WORK_FLOW = []

        self.data_split_num, self.redundancy_num, self.distributed_device_num = 0, 0, 0
        self.train_datasets, self.test_datasets = None, None
        self.base_model, self.conv_segment, self.fc_segment = None, None, None
        self.encoder = None
        self.decoder = None
        self.optimizer_encoder = None
        self.optimizer_decoder = None
        self.criterion = None
        self.scheduler = None
        self.batch_size = None
        self.current_epoch, self.epoch_num = 0, None
        self.save_interval = None
        self.save_dir = None
        self.device = None
        self.lose_device_index = None

        self._init(config)

    def run(self):
        for work in self.WORK_FLOW:
            try:
                method = getattr(self, work)
            except AttributeError:
                raise NotImplementedError(
                    f"Class `{self.__class__.__name__}` does not implement `{work}`"
                )
            method()

    def train(self):
        """
        Train the model on the train dataset.
        """
        print("Training...")
        print(
            f"split_data_num: {self.data_split_num}, "
            + f"redundancy_num: {self.redundancy_num}, "
            + f"distributed_device_num: {self.distributed_device_num}"
        )
        print("Not consider the lose device in training process.")

        self.conv_segment.to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.conv_segment.eval()
        self.encoder.train()
        self.decoder.train()

        train_loader = DataLoader(
            self.train_datasets, batch_size=self.batch_size, shuffle=False
        )

        self.loss_list = [[] for _ in range(self.epoch_num)]
        for self.current_epoch in range(self.epoch_num):
            train_loader_tqdm = tqdm(
                train_loader,
                desc=f"Epoch {self.current_epoch+1}/{self.epoch_num}",
                bar_format="{l_bar}{bar:20}{r_bar}",
            )
            correct = 0
            total = 0
            for images_list, label in train_loader_tqdm:
                images_list = [images.to(self.device) for images in images_list]
                label = label.to(self.device)

                # forward
                imageDataset_list = [
                    ImageDataset(images)
                    for images in images_list + self.encoder(images_list)
                ]
                output_list = []
                for i in range(self.distributed_device_num):
                    imageDataset = imageDataset_list[i]
                    output = self.conv_segment(imageDataset.images)
                    output_list.append(output)
                # losed_output_list = lose_something(output_list, self.lose_device_index)
                decoded_output_list = self.decoder(output_list)
                output = torch.cat(decoded_output_list, dim=3)
                output = output.view(output.size(0), -1)

                _, predicted = torch.max(self.fc_segment(output).data, 1)
                shaped_label = torch.max(self.fc_segment(label.view(label.size(0), -1)).data, 1)[1]  # FIXME: 这里的label为什么长这样？([64,16,4,4])
                correct += (predicted == shaped_label).sum().item()
                total += label.size(0)

                loss = self.criterion(output, label.view(label.size(0), -1))
                self.loss_list[self.current_epoch].append(loss.item())

                # backward
                self.optimizer_encoder.zero_grad()
                self.optimizer_decoder.zero_grad()
                loss.backward()
                self.optimizer_encoder.step()
                self.optimizer_decoder.step()

                train_loader_tqdm.set_postfix(loss=loss.item())

            print(f"Train Accuracy (Epoch {self.current_epoch+1}): {100 * correct / total}%")

        print("=" * 50)

    def test(self):
        """
        Test the model on the test dataset.
        """
        print("Testing...")
        print(
            f"split_data_num: {self.data_split_num}, "
            + f"redundancy_num: {self.redundancy_num}, "
            + f"distributed_device_num: {self.distributed_device_num}"
        )
        print(
            f"Lose Device Index: {self.lose_device_index}, "
            + f"Lose Device Num: {self.lose_device_num}"
        )

        # move models to device
        self.conv_segment.to(self.device)
        self.fc_segment.to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # set models to eval mode
        self.conv_segment.eval()
        self.fc_segment.eval()
        self.encoder.eval()
        self.decoder.eval()

        # load test dataset
        test_loader = DataLoader(
            self.test_datasets, batch_size=self.batch_size, shuffle=False
        )
        test_loader_tqdm = tqdm(
            test_loader,
            desc="Test",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        correct = 0
        total = 0
        for images_list, labels in test_loader_tqdm:
            # move data to device
            images_list = [images.to(self.device) for images in images_list]
            labels = labels.to(self.device)

            # forward
            # ----------------------------------------------
            # Encode
            imageDataset_list = [
                ImageDataset(images)
                for images in images_list + self.encoder(images_list)
            ]
            # -----------------------------------------------
            # Conv Segment
            output_list = []
            for i in range(self.distributed_device_num):
                imageDataset = imageDataset_list[i]
                output = self.conv_segment(imageDataset.images)
                output_list.append(output)
            # -----------------------------------------------
            # Lose Data
            losed_output_list = lose_something(
                output_list, self.lose_device_index, self.lose_device_num
            )
            # -----------------------------------------------
            # Decode
            decoded_output_list = self.decoder(losed_output_list)
            # -----------------------------------------------
            # FC Segment
            output = torch.cat(decoded_output_list, dim=3)
            # FIXME: output 不能无脑使用-1拉成二维数据
            output = output.view(output.size(0), -1)
            # -----------------------------------------------
            _, predicted = torch.max(self.fc_segment(output).data, 1)
            ####
            # print(predicted.shape, labels.shape) # size = ([64])
            # print(predicted, labels)
            ####
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            test_loader_tqdm.set_postfix(Accuracy=f"{correct}/{total}")

        print(f"Test Accuracy: {100 * correct / total}%")
        print("=" * 50)

    def save(self):
        """
        Save the model to the specified path.
        Saved prameters:
            - epoch: the current epoch
            - encoder_state_dict
            - decoder_state_dict
            - encoder_optimizer_state_dict
            - decoder_optimizer_state_dict
            - loss_history: [[epoch1_loss1, epoch1_loss2, ...], [epoch2_loss1, epoch2_loss2, ...], ...]
        """
        save_dir = os.path.join(
            self.save_dir,
            datetime.datetime.now().strftime("%Y_%m_%d"),
        )
        os.makedirs(save_dir, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")

        self.save_path = os.path.join(save_dir, f"train-{current_time}.pth")
        torch.save(
            {
                "epoch": self.current_epoch,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "encoder_optimizer_state_dict": self.optimizer_encoder.state_dict(),
                "decoder_optimizer_state_dict": self.optimizer_decoder.state_dict(),
                "loss_history": self.loss_list,
            },
            self.save_path,
        )
        print(f"Model saved to {self.save_path}.")

        loss_save_path = os.path.join(save_dir, f"loss-{current_time}.txt")
        with open(loss_save_path, "w") as f:
            for epoch, loss_list in enumerate(self.loss_list):
                for loss in loss_list:
                    f.write(f"epoch {epoch}: loss={loss}\n")
            f.write("\n")
            for epoch, loss_list in enumerate(self.loss_list):
                f.write(f"epoch {epoch}: avg_loss={sum(loss_list)/len(loss_list)}\n")
        print(f"Loss saved to {loss_save_path}.")

        print("=" * 50)

    def load(self):
        """
        Load the model from the specified path.
        If no load path specified, load from the last save path
        If no save path specified, raise error

        Loaded prameters:
            - epoch: the current epoch (has already done)
            - encoder_state_dict
            - decoder_state_dict
            - encoder_optimizer_state_dict
            - decoder_optimizer_state_dict
            - loss_history: [[epoch1_loss1, epoch1_loss2, ...], [epoch2_loss1, epoch2_loss2, ...], ...]
        """
        if self.load_path is None:
            if self.save_path is None:
                raise ValueError("No save path or load path specified.")
            load_path = self.save_path
        else:
            load_path = self.load_path

        checkpoint = torch.load(load_path)

        self.current_epoch = checkpoint["epoch"] + 1
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        try:
            self.optimizer_encoder.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )
            self.optimizer_decoder.load_state_dict(
                checkpoint["decoder_optimizer_state_dict"]
            )
        except:
            print("Optimizer state dict not found.")
        self.loss_list = checkpoint["loss_history"]

        print(f"Model loaded from {load_path}.")
        print("=" * 50)

    def base(self):
        """
        Test the base model on the test dataset.
        """
        # move models to device
        self.base_model.to(self.device)

        # set models to eval mode
        self.base_model.eval()

        # load test dataset
        test_loader = DataLoader(
            self.base_datasets, batch_size=self.batch_size, shuffle=False
        )
        test_loader_tqdm = tqdm(
            test_loader,
            desc="Base Model Test",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        correct = 0
        total = 0
        for img, label in test_loader_tqdm:
            # move data to device
            img = torch.cat([e.to(self.device) for e in img])
            label = label.to(self.device)

            # forward
            output = self.base_model(img)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

            test_loader_tqdm.set_postfix(Accuracy=f"{correct}/{total}")

        print(f"Baseline Accuracy: {100 * correct / total}%")
        print("=" * 50)

    def _init(self, config):
        """
        Initialize the trainer with the config.
        """
        try:
            print("loading...")

            self.WORK_FLOW = config["WORK_FLOW"]

            self._init_general(config["general"])

            for work in self.WORK_FLOW:
                init_method = getattr(self, f"_init_{work}")
                init_method(config[work])
                print(f"{str.upper(work)} parameters initialized.")

            print(f"All parameters initialized.")
            print("=" * 50)

        except KeyError as e:
            print("KeyError: {}".format(e))
            raise

    def _init_general(self, config):
        # --------------------General Parameters--------------------

        self.data_split_num = config["data_split_num"]
        self.redundancy_num = config["redundancy_num"]
        self.distributed_device_num = self.data_split_num + self.redundancy_num
        self.split_data_shape = tuple(config["split_data_shape"])

        # initialize base model
        self.base_model = construct(config["base_model"])
        self.base_model.load_state_dict(torch.load(config["base_model"]["path"]))
        self.conv_segment = self.base_model.get_conv_segment()
        self.fc_segment = self.base_model.get_fc_segment()

        # initialize encoder
        self.encoder = construct(
            config["encoder"],
            {
                "num_in": self.data_split_num,
                "num_out": self.redundancy_num,
                "in_dim": self.split_data_shape,
            },
        )

        # initialize decoder
        self.decoder = construct(
            config["decoder"],
            {
                "num_in": self.distributed_device_num,
                "num_out": self.data_split_num,
                "in_dim": self.base_model.calculate_conv_output(self.split_data_shape),
            },
        )

        # others
        self.batch_size = config["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        print(f"General parameters initialized.")

    def _init_train(self, config):
        # --------------------Training Parameters--------------------

        # import train dataset
        self.train_datasets = construct(config["train_dataset"])
        path = config["train_dataset"]["path"].replace(
            "{{ data_split_num }}", str(self.data_split_num)
        )
        self.train_datasets.load(path)
        assert self.train_datasets.split_num == self.data_split_num
        assert self.train_datasets.data_shape == self.split_data_shape
        print(f"Train dataset loaded, size: {len(self.train_datasets)}")

        # initialize optimizer
        self.optimizer_encoder = construct(
            config["encoder_optimizer"],
            {
                "params": self.encoder.parameters(),
            },
        )
        self.optimizer_decoder = construct(
            config["decoder_optimizer"],
            {
                "params": self.decoder.parameters(),
            },
        )

        # initialize criterion
        self.criterion = construct(
            config["criterion"],
            {
                "reduction": "mean",
            },
        )

        # initialize scheduler
        if config["scheduler"] is not None:
            self.scheduler = construct(
                config["scheduler"],
                {
                    "optimizer": self.optimizer_encoder,
                },
            )

        # others
        self.epoch_num = config["epoch_num"]
        self.save_interval = config["save_interval"]

    def _init_test(self, config):
        # --------------------Testing Parameters---------------------

        # test dataset
        self.test_datasets = construct(config["test_dataset"])
        path = config["test_dataset"]["path"].replace(
            "{{ data_split_num }}", str(self.data_split_num)
        )
        self.test_datasets.load(path)
        assert self.test_datasets.split_num == self.data_split_num
        assert self.test_datasets.data_shape == self.split_data_shape
        print(f"Test dataset loaded, size: {len(self.test_datasets)}")

        """
        Lose Device:        the data of "lose device" will be lost and cannot be passed to the decoder
        lose_device_index:  the index of the device, eg. (0, 2,)
        lose_device_num:    the number of data that will be lost, eg. 2,
                            which means random 2 data will be lost.
                            lose_device_num will work only when lose_device_index is None.
        """
        self.lose_device_index = config["lose_device_index"]
        self.lose_device_num = config["lose_device_num"]

    def _init_save(self, config):
        # --------------------Save Parameters------------------------

        # save dir
        self.save_dir = config["save_dir"]

    def _init_load(self, config):
        # --------------------Load Parameters------------------------

        # load dir
        self.load_path = config["load_path"]

    def _init_base(self, config):
        self.base_datasets = construct(config["dataset"])
        self.base_datasets.load(config["dataset"]["path"])
