from main import *
import time
from torch.utils.data import DataLoader

# 定义训练的设备
device = torch.device("cuda")

# 数据增强
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 可选：进行归一化处理
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=train_transform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=train_transform,
                                          download=True)

# 训练集和测试集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集长度为：{train_data_size}")
print(f"测试数据集长度为：{test_data_size}")

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建网络模型
model = VisionNet()

model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 定义学习率调度器（每10个epoch学习率减半）
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 50

# 添加tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./logs_train")
cumulative_test_loss = 0
cumulative_accuracy = 0
num_test_batches = len(test_dataloader)

start_time = time.time()

for i in range(epoch):
    print(f"\n————————————第{i+1}轮训练开始————————————")

    # 训练步骤开始
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f"训练次数：{total_train_step}, Loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 计算正确预测的样本数
        correct = (outputs.argmax(1) == targets).sum().item()
        total_train_correct += correct

        # 累加训练损失
        total_train_loss += loss.item()

    # 计算训练集上的平均Loss和正确率
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_correct / len(train_dataloader.dataset)

    # 记录训练集上的平均Loss和正确率到TensorBoard
    writer.add_scalar("train_loss_avg", avg_train_loss, total_train_step)
    writer.add_scalar("train_accuracy_avg", avg_train_accuracy, total_train_step)

    # 更新学习率
    scheduler.step()

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # 累加每个批次的Loss和正确率
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy +=accuracy

        # 增加测试步骤计数器
        total_test_step += 1
        # 计算测试集的平均Loss和正确率
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_accuracy = total_accuracy / len(test_dataloader.dataset)
        # 记录测试集的平均Loss和正确率到TensorBoard
        writer.add_scalar("test_loss_avg", avg_test_loss, total_test_step)
        writer.add_scalar("test_accuracy_avg", avg_accuracy, total_test_step)
        # 重置累计值，为下一个epoch做准备
        cumulative_test_loss = 0
        cumulative_accuracy = 0
        # 打印整体测试集上的Loss和正确率
        print(f"\n整体测试集上的平均Loss: {avg_test_loss}")
        print(f"整体测试集上的平均正确率: {avg_accuracy}")

    # 保存模型
    torch.save(model.state_dict(), f"./model_i/model_{i+1}.pth")
    print("模型已保存")

writer.close()
