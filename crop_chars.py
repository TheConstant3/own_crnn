import sys

import cv2
import numpy as np


def crop_from_image():
        image = cv2.imread('templates/BY-auto-number-1.png', cv2.IMREAD_UNCHANGED)
        trans_mask = image[:, :, 3] == 0

        new_img = image.copy()

        #replace areas of transparency with white and not transparent
        new_img[trans_mask] = [255, 255, 255, 255]

        #new image without alpha channel...
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGRA2BGR)

        # new_img = cv2.blur(new_img, (3, 3,))
        bin_img = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY)[1]

        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789aou'

        now_chars = False

        y_crop_1 = 0
        y_crop_2 = 0
        x_crop_1 = 0
        x_crop_2 = 0

        cropped_rows = []

        i = 0

        for y in range(bin_img.shape[0]):
            print(np.mean(bin_img[y, :]))
            if np.mean(bin_img[y, :]) < 255:
                if not now_chars:
                    y_crop_1 = y
                    now_chars = True
            elif now_chars:
                y_crop_2 = y
                now_chars = False
                bin_row = bin_img[y_crop_1:y_crop_2, :]

                # cv2.imshow('', bin_row)
                # cv2.waitKey(0)

                for x in range(bin_row.shape[1]):
                    if np.mean(bin_row[:, x]) < 255:
                        if not now_chars:
                            x_crop_1 = x
                            now_chars = True
                    elif now_chars:
                        x_crop_2 = x
                        now_chars = False


                        # cv2.imshow(chars[i], bin_row[:, x_crop_1:x_crop_2])
                        # cv2.waitKey(0)
                        cv2.imwrite(f'templates/chars_/{chars[i]}.png', image[y_crop_1:y_crop_2, x_crop_1:x_crop_2])
                        i += 1
                        if i == len(chars):
                            sys.exit(0)

img = cv2.imread('templates/chars/_.png', cv2.IMREAD_UNCHANGED)
bin_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
bin_img = cv2.threshold(bin_img, 100, 255, cv2.THRESH_BINARY)[1]

trans_mask = bin_img[:, :] == 255

print(trans_mask.shape, img.shape)

#replace areas of transparency with white and not transparent
img[trans_mask] = [255, 255, 255, 0]

cv2.imwrite('templates/chars/_.png', img)

# cv2.imshow('', bin_img)
# cv2.waitKey(0)

