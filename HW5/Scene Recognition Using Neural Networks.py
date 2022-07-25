import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt


"""##**use gpu if it's available**"""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = {
    '''
    note that all files are saved in folders.
    Pictures of the same category are stored in each folder. 
    The folder name is class name

    '''

    'extracted_root': "/content/Data",  # path of extracted zip data
    'train_data_path': "Data/Train",  # path of extracted train data
    'test_data_path': "Data/Test",  # path of extracted test data
    'batch_size': 64,
    'num_workers': 0,
    'num_classes': 15,
    'eopches': 50,
    'learning_rate': 0.0005
}


# define normalization transform
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],)


# define train_set transform
train_transform1 = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize, ])

# --------------------------

train_transform2 = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    normalize, ])
# #---------------------------

# define test_set transform
test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize, ])

"""##**split data augmentation**
one good way to split data and pass it to `dataloader` is to use `torchvision.datasets.ImageFolder` . this function get images with their label without using any label file . `Imagefolder` assumes that all files are saved in folders ,Pictures of the same category are stored in each folder and The folder name is class name .

 we also use `torch.utils.data.ConcatDataset` for concatenating the new training dataset created by `transforms.RandomHorizontalFlip(p=1)` with original training dataset

"""

# ---trainig set
all_datasets = []
train_data_aug1 = torchvision.datasets.ImageFolder(
    root=config['train_data_path'], transform=train_transform1)
all_datasets.append(train_data_aug1)
train_data_aug2 = torchvision.datasets.ImageFolder(
    root=config['train_data_path'], transform=train_transform2)
all_datasets.append(train_data_aug2)

train_data_total = torch.utils.data.ConcatDataset(all_datasets)
train_data_loader = data.DataLoader(
    train_data_total, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

# ---test set
test_data = torchvision.datasets.ImageFolder(
    root=config['test_data_path'], transform=test_transform)
test_data_loader = data.DataLoader(
    test_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

"""length of train dataset before and after augmentation"""

print(f"length of train_set before augmentation :  {len(train_data_aug1)}")
print(f"length of train_set after augmentation : {len(train_data_total)}")

"""##**define funtions**

* measure accuracy
"""


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


"""* train and evaluate the accuracy/loss in each epoch"""


def train_model(model):

    # ------------------------------------
    total_step_test = len(test_data_loader)
    # ----------    train   --------------
    total_step_train = len(train_data_loader)
    # ------------------------------------
    loss_train = []
    acc1_train = []
    acc5_train = []
    # -----------------
    loss_test = []
    acc1_test = []
    acc5_test = []

    for epoch in range(config['eopches']):

        sum_of_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0

        for i, (images, labels) in enumerate(train_data_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # --Forward pass
            outputs = model(images)
            # ----measure acc
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            sum_acc1 += float(acc1)
            sum_acc5 += float(acc5)
            # ----measure loss
            loss = criterion(outputs, labels)
            sum_of_loss += float(loss)
            #----Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_of_each_epoch = float(sum_of_loss/total_step_train)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch+1, config['eopches'], i+1, total_step_train, loss_of_each_epoch))

        print('Acc@1 of train_set: {} %'.format(sum_acc1 / total_step_train))

        loss_train.append(loss_of_each_epoch)
        acc1_train.append(sum_acc1 / total_step_train)
        acc5_train.append(sum_acc5 / total_step_train)

        # measure acc on test_set                #-------------------------------------
        # -------------  test  ----------------
        with torch.no_grad():  # -------------------------------------

            sum_acc1 = 0
            sum_acc5 = 0
            sum_of_loss = 0

            for i, (images, labels) in enumerate(test_data_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # measure loss
                loss = criterion(outputs, labels)
                sum_of_loss += float(loss)

                # measure acc
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                sum_acc1 += float(acc1)
                sum_acc5 += float(acc5)

                del images, labels, outputs
            loss_of_each_epoch = float(sum_of_loss/total_step_test)

            loss_test.append(loss_of_each_epoch)
            acc1_test.append(sum_acc1 / total_step_test)
            acc5_test.append(sum_acc5 / total_step_test)

            print('Acc@1 of validation_set: {} %'.format(sum_acc1 / total_step_test))
            print('------------------------------')

    return loss_test, acc1_test, acc5_test, loss_train, acc1_train, acc5_train


"""* plot function for acc@1 , acc@5 and loss with respect to epoches"""


def acc_loss_plot(loss_test, acc1_test, acc5_test, loss_train, acc1_train, acc5_train):
    # define horizental axis
    x_axis = np.array([i for i in np.arange(1, 51)])

    # ---plot loss function for train and test set in each epoch
    plt.plot(x_axis, loss_train, label="trian loss")
    plt.plot(x_axis, loss_test, label="test loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss on train and test set")
    plt.show()
    # --------------------------------

    # ---plot acc@1 for train and test set in each epoch
    plt.plot(x_axis, acc1_train, label="trian accuracy")
    plt.plot(x_axis, acc1_test, label="test accuracy")
    plt.legend(loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("acc@1 on train and test set")

    plt.show()

    # ---plot acc@5 for train and test set in each epoch
    plt.plot(x_axis, acc5_train, label="trian accuracy")
    plt.plot(x_axis, acc5_test, label="test accuracy")
    plt.legend(loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("acc@5 on train and test set")

    plt.show()


# **part 1**

class AlexNet(nn.Module):
    def __init__(self, num_classes=15):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            # nn.BatchNorm2d(96),
            nn.LocalResponseNorm(2),  # adding new additional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16224, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


"""###**load model and optimizer**"""

# load the model
model = AlexNet(config['num_classes']).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=config['learning_rate'], momentum=0.9)

"""###**train the model**"""

loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr = train_model(model)

"""##**plot loss and accuracy**"""

acc_loss_plot(loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr)


# **part 2**

class AlexNet(nn.Module):
    def __init__(self, num_classes=15):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            # nn.BatchNorm2d(96),
            nn.LocalResponseNorm(2),  # adding new additional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# **load model**"""


# load the model
model = AlexNet(config['num_classes']).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=config['learning_rate'], momentum=0.9)

"""###**train model**"""

loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr = train_model(model)

"""###**plot loss and accuracy**"""

acc_loss_plot(loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr)

"""$Results  : $


$maximum \ acc@1 \  on \  test \ set \ converges \  to  \ : \ 73.91  $%

$maximum \ acc@5 \  on \  test \ set \ converges \  to  \ : \ 93.5  $%


$maximum \ acc@1 \  on \  trian \ set \ converges \  to  \ : \ 97.27  $%

$maximum \ acc@5 \  on \  train \ set \ converges \  to  \ : \ 100  $%


---

$conclusions  : $

as we can see there is no overfiting in training model , in addition the curve of loss-vs-epoch  Meet our expectations because at first , loss of test is bigger than train and over epoches , the train loss becomes less and less but test loss becomes approximately constant after the 20th epoch . there is also a gap between test loss and train loss after some epoches. 
also the curve of acc@1-vs-epoch  Meet our expectations because there is a gap between test's acc@1 and train's acc@1 over epoches.

###**part 3**


whole alexnet arcitecture without using trained weights
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes=15):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


"""###**load model**"""

# load the model
model = AlexNet(config['num_classes']).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=config['learning_rate'], momentum=0.9)

"""###**train model**"""

loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr = train_model(model)

"""###**plot loss and accuracy**"""

acc_loss_plot(loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr)

"""$Results  : $

$maximum \ acc@1 \  on \  test \ set \ converges \  to  \ : \ 76.64  $%

$maximum \ acc@5 \  on \  test \ set \ converges \  to  \ : \ 96  $%


$maximum \ acc@1 \  on \  trian \ set \ converges \  to  \ : \ 99.47  $%

$maximum \ acc@5 \  on \  train \ set \ converges \  to  \ : \ 100  $%

---

$conclusions  : $

same as previous parts

###**part 4**
whole alexnet architecture : 
we squeeze the pre-trained weights ad just learn the new weight of classifier layer :
"""

AlexNet_model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
# ---description
AlexNet_model.eval()

"""###**freeze layers before classifier**"""

AlexNet_model.classifier[6] = nn.Linear(4096, 15)
AlexNet_model.eval()

for param in AlexNet_model.parameters():
    param.requires_grad = False  # freeze all layers
for param in AlexNet_model.classifier[6].parameters():
    param.requires_grad = True  # unfreez classifer layer

"""###**load the model**



"""

# load the model
model = AlexNet_model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=config['learning_rate'], momentum=0.9)

"""###**train model**"""

loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr = train_model(model)

"""###**plot loss and accuracy**"""

acc_loss_plot(loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr)

"""$Results  : $

$maximum \ acc@1 \  on \  test \ set \ converges \  to  \ : \ 89.2  $%

$maximum \ acc@5 \  on \  test \ set \ converges \  to  \ : \ 99.8  $%


$maximum \ acc@1 \  on \  trian \ set \ converges \  to  \ : \ 99.86  $%

$maximum \ acc@5 \  on \  train \ set \ converges \  to  \ : \ 100  $%

---

$conclusions  : $

same as previous parts

###**part 5**
transfer learning : 
in this part we use trained weights of alexnet model that use for imagenet dataset
"""

AlexNet_model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
# ---description
AlexNet_model.eval()

"""###change last fully connected layers of this arcitecture  

"""

AlexNet_model.classifier[6] = nn.Linear(4096, 15)

# ---description of new model
AlexNet_model.eval()

"""###**load model**"""

# load the model
model = AlexNet_model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer(SGD)
optimizer = torch.optim.SGD(
    AlexNet_model.parameters(), lr=config['learning_rate'], momentum=0.9)

"""###**train the model**"""

loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr = train_model(model)

"""###**plot loss and accuracy**"""

acc_loss_plot(loss_te, acc1_te, acc5_te, loss_tr, acc1_tr, acc5_tr)

"""$Results  : $

$maximum \ acc@1 \  on \  test \ set \ converges \  to  \ : \ 89.43  $%

$maximum \ acc@5 \  on \  test \ set \ converges \  to  \ : \ 99.7  $%


$maximum \ acc@1 \  on \  trian \ set \ converges \  to  \ : \ 99.93  $%

$maximum \ acc@5 \  on \  train \ set \ converges \  to  \ : \ 100  $%

---

$conclusions  : $

same as previous parts
"""
