import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(in_features=32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.fc6 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def calculate_metrics(loader, model):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, recall, precision, f1


transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform)
valset = torchvision.datasets.SVHN(root='./data', split="test", download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
valloader = DataLoader(valset, batch_size=128, shuffle=False)


net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

plt.ion()

tr_acc = []
tr_recall = []
tr_prec = []
tr_f1 = []
vali_acc = []
vali_recall = []
vali_prec = []
vali_f1 = []


for epoch in range(30):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

    train_acc, train_rec, train_prec, train_f1 = calculate_metrics(trainloader, net)
    print(f'Training - Accuracy: {train_acc}, Recall: {train_rec}, Precision: {train_prec}, F1 Score: {train_f1}')

    val_acc, val_rec, val_prec, val_f1 = calculate_metrics(valloader, net)
    print(f'Validation - Accuracy: {val_acc}, Recall: {val_rec}, Precision: {val_prec}, F1 Score: {val_f1}')

    x = [i for i in range(1, epoch + 2)]

    tr_acc.append(train_acc)
    tr_recall.append(train_rec)
    tr_prec.append(train_prec)
    tr_f1.append(train_f1)
    vali_acc.append(val_acc)
    vali_recall.append(val_rec)
    vali_prec.append(val_prec)
    vali_f1.append(val_f1)

    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title('Accuracy', fontsize=10)
    plt.plot(x, tr_acc, 'b', label='training')
    plt.plot(x, vali_acc, 'r', label='validation')

    plt.subplot(2, 2, 2)
    plt.title('Recall', fontsize=10)
    plt.plot(x, tr_recall, '--b', label='training')
    plt.plot(x, vali_recall, '--r', label='validation')

    plt.subplot(2, 2, 3)
    plt.title('Precision', fontsize=10)
    plt.plot(x, tr_prec, '-.b', label='training')
    plt.plot(x, vali_prec, '-.r', label='validation')

    plt.subplot(2, 2, 4)
    plt.title('F1 Score', fontsize=10)
    plt.plot(x, tr_f1, ':b', label='training')
    plt.plot(x, vali_f1, ':r', label='validation')

    plt.draw()
    plt.gcf().canvas.flush_events()

plt.ioff()
plt.show()
print('Finished Training')