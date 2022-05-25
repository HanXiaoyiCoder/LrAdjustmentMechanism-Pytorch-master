from torch.utils.data import DataLoader
from torchvision.models import AlexNet
from torchvision import transforms
import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

# 1.Create SummaryWriter
writer = SummaryWriter("tensorboard")

# 2.Ready dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=transforms.Compose(
    [transforms.Resize(227), transforms.ToTensor()]), download=True)

print('CUDA available: {}'.format(torch.cuda.is_available()))

# 3.Length
train_dataset_size = len(train_dataset)
print("the train dataset size is {}".format(train_dataset_size))

# 4.DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)

# 5.Create model
model = AlexNet()

if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)

# 6.Create loss
cross_entropy_loss = nn.CrossEntropyLoss()

# 7.Optimizer
learning_rate = 1e-3
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5, 10, 15], gamma=0.1, verbose=True)

# 8. Set some parameters to control loop
# epoch
epoch = 20

iter = 0
t0 = time.time()
for i in range(epoch):
    t1 = time.time()
    print(" -----------------the {} number of training epoch --------------".format(i))
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            cross_entropy_loss = cross_entropy_loss.cuda()
            imgs, targets = imgs.cuda(), targets.cuda()
        outputs = model(imgs)
        loss_train = cross_entropy_loss(outputs, targets)
        writer.add_scalar("train_loss", loss_train.item(), iter)
        optim.zero_grad()
        loss_train.backward()
        optim.step()
        iter = iter + 1
        if iter % 100 == 0:
            print(
                "Epoch: {} | Iteration: {} | lr1: {} | lr2: {} |loss: {} | np.mean(loss): {} "
                    .format(i, iter, scheduler.get_lr()[0], scheduler.get_last_lr()[0], loss_train.item(),
                            np.mean(loss_train.item())))

    writer.add_scalar("lr", scheduler.get_lr()[0], i)
    writer.add_scalar("lr_last", scheduler.get_last_lr()[0], i)
    scheduler.step()
    t2 = time.time()
    h = (t2 - t1) // 3600
    m = ((t2 - t1) % 3600) // 60
    s = ((t2 - t1) % 3600) % 60
    print("epoch {} is finished, and time is {}h{}m{}s".format(i, int(h), int(m), int(s)))

    if i % 1 == 0:
        print("Save state, iter: {} ".format(i))
        torch.save(model.state_dict(), "checkpoint/AlexNet_{}.pth".format(i))

torch.save(model.state_dict(), "checkpoint/AlexNet.pth")
t3 = time.time()
h_t = (t3 - t0) // 3600
m_t = ((t3 - t0) % 3600) // 60
s_t = ((t3 - t0) % 3600) // 60
print("The finished time is {}h{}m{}s".format(int(h_t), int(m_t), int(s_t)))
writer.close()
