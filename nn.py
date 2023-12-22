#!/usr/bin/env python3
"""Train neural network"""

import pickle
import os
import copy
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

def main():
    data_dir = "/Users/kylecoon/Desktop/ChessProject/data_large"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def get_dataloaders(input_size, batch_size, shuffle = True):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        composed_transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

        data_transforms = {
            'train': composed_transform,
            'val': composed_transform,
            'test': composed_transform
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}

        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=2) for x in data_transforms.keys()}

        return dataloaders_dict

    batch_size = 16
    input_size = 256
    dataloaders_dict = get_dataloaders(input_size, batch_size)

    print('# of training samples {}'.format(len(dataloaders_dict['train'].dataset)))
    print('# of validation samples {}'.format(len(dataloaders_dict['val'].dataset)))
    print('# of test samples {}'.format(len(dataloaders_dict['test'].dataset)))

    def count_params(model):

        num_params = sum([item.numel() for item in model.parameters() if item.requires_grad])

        return num_params
    
    cfgs = {
        'VGG': [64, 'M', 128, 'M', 128, 128, 'M'],
        'VGG-BN': [64, 'M', 128, 'M', 128, 128, 'M']
    }

    def make_layers(cfg, batch_norm=False):

        layers = []
        channels = 3

        for i in range(len(cfg)):
            if cfg[i] == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(channels, cfg[i], kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                channels = cfg[i]

        features = nn.Sequential(*layers)

        return features

    class VGG(nn.Module):

        def __init__(self, features, num_classes=13, init_weights=True):
            super(VGG, self).__init__()

            self.features = features
            self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

            self.classifier = nn.Sequential(
                nn.Linear(128 * 5 * 5, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
                )

            if init_weights:
                self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    features = make_layers(cfgs['VGG'], batch_norm=False)
    vgg = VGG(features)

    features = make_layers(cfgs['VGG-BN'], batch_norm=True)
    vgg_bn = VGG(features)

    print(vgg)
    print('Number of trainable parameters {}'.format(count_params(vgg)))

    print(vgg_bn)
    print('Number of trainable parameters {}'.format(count_params(vgg_bn)))

    def make_optimizer(model):
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        return optimizer

    def get_loss():
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def train_model(model, dataloaders, criterion, optimizer, save_dir = None, num_epochs=25, model_name='VGG'):

        val_acc_history = []
        tr_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval() 

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if phase == 'train':
                        optimizer.zero_grad()

                    torch.set_grad_enabled(phase == 'train')
                    outputs = model(inputs)
                    vals, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    if save_dir:
                        torch.save(best_model_wts, os.path.join(save_dir, model_name + '.pth'))

                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                else:
                    tr_acc_history.append(epoch_acc)

        print('Best val Acc: {:4f}'.format(best_acc))

        return model, tr_acc_history, val_acc_history

    batch_size = 16

    shuffle_datasets = True

    num_epochs = 10

    save_dir = "/Users/kylecoon/Desktop/ChessProject/weights"
    os.makedirs(save_dir, exist_ok=True)

    input_size = 256
    dataloaders = get_dataloaders(input_size, batch_size, shuffle_datasets)
    criterion = get_loss()

    features = make_layers(cfgs['VGG'], batch_norm=False)
    model = VGG(features).to(device)
    optimizer = make_optimizer(model)

    vgg, tr_his, val_his = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
            save_dir=save_dir, num_epochs=num_epochs, model_name='VGG')
    
    features = make_layers(cfgs['VGG-BN'], batch_norm=True)
    model = VGG(features).to(device)
    optimizer = make_optimizer(model)

    vgg_BN, tr_his_BN, val_his_BN = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
            save_dir=save_dir, num_epochs=num_epochs, model_name='VGG-BN')
    
    pickle.dump(tr_his, open('tr_his.pkl', 'wb'))
    pickle.dump(tr_his_BN, open('tr_his_BN.pkl', 'wb'))
    pickle.dump(val_his, open('val_his.pkl', 'wb'))
    pickle.dump(val_his_BN, open('val_his_BN.pkl', 'wb'))

    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def test(model, dataloader):

        model.eval()

        top1_acc = []
        top5_acc = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                res = accuracy(outputs, labels, topk=(1, 5))

                top1_acc.append(res[0] * len(outputs))
                top5_acc.append(res[1] * len(outputs))

        print('Top-1 accuracy {}%, Top-5 accuracy {}%'.format(sum(top1_acc).item()/303, sum(top5_acc).item()/303))

    vgg_BN.load_state_dict(torch.load('/Users/kylecoon/Desktop/ChessProject/weights/VGG-BN.pth'))
    vgg.load_state_dict(torch.load('/Users/kylecoon/Desktop/ChessProject/weights/VGG.pth'))

    test(vgg_BN, dataloaders['test'])
    test(vgg, dataloaders['test'])
    
if __name__ == "__main__":
    main()