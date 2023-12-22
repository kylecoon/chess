#!/usr/bin/env python3
"""Format Input"""

import numpy as np, matplotlib.pyplot as os
import imageio
import imageio.v2 as imageio
import math
import cv2


def main():
    """Main thread, which reads input and normalizes it"""

    directory = '/Users/kylecoon/Desktop/ChessProject/input_images/'
    files = ['board_1.jpg', 'board_2.jpg', 'board_3.jpg', 'board_4.jpg', 'board_5.jpg', 'board_6.jpg', 'board_7.jpg', 'board_8.jpg', 'board_9.jpg', 'board_10.jpg', 'board_11.jpg', 'board_12.jpg', 'board_13.jpg', 'board_14.jpg', 'board_15.jpg', 'board_16.jpg', 'board_17.jpg', 'board_18.jpg', 'board_19.jpg', 'board_20.jpg', 'board_21.jpg', 'board_22.jpg', 'board_23.jpg', 'board_24.jpg', 'board_25.jpg', 'board_26.jpg', 'board_27.jpg', 'board_28.jpg', 'board_29.jpg', 'board_30.jpg', 'board_31.jpg', 'board_32.jpg']
    for filename in files:
        # read input
        f = os.path.join(directory, filename)
        im = cv2.imread(f)
        im = cv2.resize(im, (2048, 2048), interpolation=cv2.INTER_AREA)

        #gray scale
        bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        #blur
        kernel = np.ones((31,31),np.float32)/961
        binarized  = cv2.filter2D(bw,-1,kernel)

        #binarize
        threshold = 128
        for i in range(binarized.shape[0]):
            for j in range(binarized.shape[1]):
                tmp = im[i][j][0]
                im[i][j][0] = im[i][j][2]
                im[i][j][2] = tmp
                if i > 512 and i < 1536 and j > 512 and j < 1536:
                    binarized[i][j] = 0
                else:
                    if binarized[i][j] > threshold:
                        binarized[i][j] = 255
                    else:
                        binarized[i][j] = 0

        #find corners
        corners = cv2.goodFeaturesToTrack(binarized, 500, 0.01, 50)

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

        cv2.circle(im, topLeft, 51, (255, 0, 0), -1)
        cv2.circle(im, topRight, 51, (255, 0, 0), -1)
        cv2.circle(im, bottomLeft, 51, (255, 0, 0), -1)
        cv2.circle(im, bottomRight, 51, (255, 0, 0), -1)

        #transform to normalized square input
        foundPoints = np.float32([topLeft, bottomLeft, topRight, bottomRight])
        dstPoints = np.float32([(0,0), (2047,0), (0,2047), (2047,2047)])

        M = cv2.getPerspectiveTransform(foundPoints, dstPoints)
        transformed = cv2.warpPerspective(im, M, (2048,2048))

        #zoom in a bit to account for my chess board
        zoomPoints = np.float32([(138, 138), (1909, 138), (138, 1909), (1909, 1909)])
        M = cv2.getPerspectiveTransform(zoomPoints, dstPoints)
        transformed = cv2.warpPerspective(transformed, M, (2048,2048))                    

        #output individual squares
        count = 1
        filename = filename.split('.')[0]
        for i in range(0, 2048, 256):
            for j in range(0, 2048, 256):
                cropped = transformed[i:i+256, j:j+256]
                imageio.imwrite(f'/Users/kylecoon/Desktop/ChessProject/output_images/{filename}_{count}.jpg', cropped)
                count += 1

if __name__ == "__main__":
    main()