import numpy as np
import cv2
import matplotlib.pyplot as plt


def display(img, cmap=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()


coins = cv2.imread('../data/coins.png')
display(coins)

coins = cv2.medianBlur(coins, 35)
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
display(thresh, cmap='gray')

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
display(opening, cmap='gray')

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
display(sure_bg, cmap='gray')

# finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
display(dist_transform, cmap='gray')
display(sure_fg, cmap='gray')

# finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
display(unknown, cmap='gray')

# marker labeling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
display(markers, cmap='gray')

# apply Watershed Algorithm to find markers
markers = cv2.watershed(coins, markers)
display(markers)

# finding contours on markers
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coins, contours, i, (255, 0, 0), 10)

display(coins)
