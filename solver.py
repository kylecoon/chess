#!/usr/bin/env python3
"""Solve one image"""

import numpy as np
import imageio.v2 as imageio
import math
import cv2
import torch

import torch.nn as nn

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

def main():
    input_image_name = "/Users/kylecoon/Desktop/ChessProject/input_images/board_4.jpg"

    input_image = cv2.imread(input_image_name)
    input_image = cv2.resize(input_image, (2048, 2048), interpolation=cv2.INTER_AREA)

    #gray scale
    bw = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    #blur
    kernel = np.ones((31,31),np.float32)/961
    binarized  = cv2.filter2D(bw,-1,kernel)

    #binarize
    threshold = 128
    for i in range(binarized.shape[0]):
        for j in range(binarized.shape[1]):
            tmp = input_image[i][j][0]
            input_image[i][j][0] = input_image[i][j][2]
            input_image[i][j][2] = tmp
            if i > 512 and i < 1536 and j > 512 and j < 1536:
                binarized[i][j] = 0
            else:
                if binarized[i][j] > threshold:
                    binarized[i][j] = 255
                else:
                    binarized[i][j] = 0

    #find corners
    corners = cv2.goodFeaturesToTrack(binarized, 500, 0.01, 10)

    topLeft = (-1, -1)
    topRight = (-1, -1)
    bottomLeft = (-1, -1)
    bottomRight = (-1, -1)


    comps = [(0, 0), (0,2047), (2047,0), (2047,2047)]
    for comp in comps:
        minDistance = 1000
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            if math.dist((x, y), comp) < minDistance:
                minDistance = math.dist((x, y), comp)
                    #find topleft
                if comp == (0, 0):
                    topLeft = (x, y)
                    #find topright
                if comp == (0, 2047):
                    topRight = (x, y)
                    #find bottomleft
                if comp == (2047, 0):
                    bottomLeft = (x, y)
                    #find bottomright
                if comp == (2047, 2047):
                    bottomRight = (x, y)


        #transform to normalized square input
    foundPoints = np.float32([topLeft, bottomLeft, topRight, bottomRight])
    dstPoints = np.float32([(0,0), (2047,0), (0,2047), (2047,2047)])

    M = cv2.getPerspectiveTransform(foundPoints, dstPoints)
    transformed = cv2.warpPerspective(input_image, M, (2048,2048))

        #zoom in a bit to account for my chess board border
    zoomPoints = np.float32([(138, 138), (1909, 138), (138, 1909), (1909, 1909)])
    M = cv2.getPerspectiveTransform(zoomPoints, dstPoints)
    transformed = cv2.warpPerspective(transformed, M, (2048,2048))

    #create array of individual squares
    squares = []
    for i in range(0, 2048, 256):
        for j in range(0, 2048, 256):
            squares.append(transformed[i:i+256, j:j+256])
    
    fen = ""
    features = make_layers([64, 'M', 128, 'M', 128, 128, 'M'], batch_norm=True)
    model = VGG(features)
    model.load_state_dict(torch.load("/Users/kylecoon/Desktop/ChessProject/weights/VGG-BN.pth"))
    model.eval()

    classes_dict =  {0 : "1",
                    1 : "b",
                    2 : "k",
                    3 : "n",
                    4 : "p",
                    5 : "q",
                    6 : "r",
                    7 : "B",
                    8 : "K",
                    9 : "N",
                    10 : "P",
                    11 : "Q",
                    12 : "R"
                }
    points_dict =  {0 : 0,
               1 : -3,
               2 : 0,
               3 : -3,
               4 : -1,
               5 : -9,
               6 : -5,
               7 : 3,
               8 : 0,
               9 : 3,
               10 : 1,
               11 : 9,
               12 : 5
                }
    count = 0
    points = 0
    for square in squares:
        square_tensor = torch.from_numpy(square).permute(2, 0, 1).float().unsqueeze(0)

        results = model(square_tensor)
        predicted_class = torch.argmax(results, dim=1).item()

        points += points_dict[predicted_class]
        if (count % 8 == 0):
            fen += '/'
        if fen[len(fen)-1].isnumeric() and classes_dict[predicted_class] == "1":
            newNum = str(int(fen[len(fen)-1]) + 1)
            fen = fen[:-1]
            fen += newNum
        else:
            fen += classes_dict[predicted_class]
        count += 1

    if points > 0:
        print("\nWhite is winning by", points, "points.\n")
    elif points < 0: 
        print("\nBlack is winning by", abs(points), "points.\n")
    else:
        print("\nThe game is even!\n")
    print("Want to analyze further? Check out https://lichess.org/analysis/standard" + fen)

if __name__ == "__main__":
    main()