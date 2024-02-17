import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main(rank, world_size):
    # 设置设备
    device = torch.device(f"cuda:{rank}")

    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建模型
    model = SimpleModel().to(device)
    # 将模型封装为分布式数据并行模型
    ddp_model = DDP(model, device_ids=[rank])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 数据
    inputs = torch.randn(20, 10).to(device)
    labels = torch.randint(0, 2, (20,)).to(device)

    # 训练
    for epoch in range(100):
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item()}")

    # 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'  # 主节点的地址
    os.environ['MASTER_PORT'] = '12345'  # 主节点的端口

    # 设置分布式训练的参数
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
