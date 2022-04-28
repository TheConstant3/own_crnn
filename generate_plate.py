import concurrent.futures
import os
import random
import time
from functools import partial
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
import albumentations.augmentations as A

chars_dir = 'templates/chars'
fonts_dir = 'templates/fonts'
bkgs_dir = 'templates/bkgs'


def generate_with_fonts():

    hex_color_am = 'E5EAF0'

    color = tuple([int(hex_color_am[i:i+2], 16) for i in (0, 2, 4)] + [255])
    for font in os.listdir(fonts_dir)[3:]:
        if font == 'Eurostile-Bold Regular.ttf':
            size = 55
        else:
            size = 65
        # get an image
        with Image.open('templates/bkgs/eec-german-license-plate.jpg').convert("RGBA") as base:
            # get a font
            fnt = ImageFont.truetype(f"{fonts_dir}/{font}", size)

            # make a blank image for the text, initialized to transparent text color
            txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
            # get a drawing context
            d = ImageDraw.Draw(txt)

            d.rectangle((40, 10, 340, 70), fill=color)
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            # draw text, half opacity
            # d.text((38, 15), "HELL O0O0", font=fnt, fill=(0, 25, 25, 255))
            d
            out = Image.alpha_composite(base, txt)

            cv2.imshow(font, cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGRA))
            break
    cv2.waitKey(0)


def get_char_images():
    char_image_dict = dict()
    for char_file in os.listdir(chars_dir):
        value = char_file.split('.')[0]
        char_image_dict[value] = cv2.imread(f'{chars_dir}/{char_file}', cv2.IMREAD_UNCHANGED)
    return char_image_dict


def imshow(win_name, image_orig):
    image = image_orig.copy()
    if image.shape[2] == 4:
        _, mask = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        image[mask == 0] = (255, 255, 255, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imshow(win_name, image)
    cv2.waitKey(0)


def get_bkg_plate_images():
    fill_color = (240, 240, 240)
    bkg_plates = list()
    for bkg_file in os.listdir(bkgs_dir):
        bkg_image = cv2.imread(f'{bkgs_dir}/{bkg_file}')
        # print(bkg_file, bkg_image.shape)
        bkg_image[7:70, 40:340] = fill_color
        bkg_image = cv2.cvtColor(bkg_image, cv2.COLOR_BGR2BGRA)
        bkg_plates.append(bkg_image)
    return bkg_plates


def increase_bkg(image):
    h, w = image.shape[:2]
    bkg = np.zeros((h * 2, w * 2, 4), dtype=np.uint8)
    x_paste = w // 2
    y_paste = h // 2
    bkg[y_paste:y_paste + h, x_paste:x_paste + w] = image
    return bkg


def resize_char(image):
    h, w = image.shape[:2]
    scale = 1 - random.random() * 0.3
    h = int(h * scale)
    w = int(w * scale)
    return cv2.resize(image, (w, h))


def rotate_char(image, rotate_limit=20):
    h, w = image.shape[:2]
    scale = 1 - random.random() * 0.3
    angle = int(random.random() * rotate_limit * 2 - rotate_limit)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    return cv2.warpAffine(image, M, (w, h))


def change_char_color(image):
    _, mask = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    new_color = [random.randint(0, 30) if i < 3 else 255 for i in range(4)]
    image[mask > 0] = new_color
    return image


def crop_without_bkg(image):
    _, mask = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    x1, x2, y1, y2 = 0, 0, 0, 0

    for i in range(mask.shape[1]):
        x_row = np.mean(mask[:, i])
        if x_row > 0:
            x1 = i
            break
    for i in reversed(range(mask.shape[1])):
        x_row = np.mean(mask[:, i])
        if x_row > 0:
            x2 = i
            break

    for i in range(mask.shape[0]):
        y_row = np.mean(mask[i, :])
        if y_row > 0:
            y1 = i
            break
    for i in reversed(range(mask.shape[0])):
        y_row = np.mean(mask[i, :])
        if y_row > 0:
            y2 = i
            break
    return image[y1:y2, x1:x2]


def preprocess_char_image(max_height, foreground, stage):
    if foreground is None:
        return None

    resized = False
    if random.random() > (1 - (stage * 0.2)):
        foreground = resize_char(foreground)
        resized = True

    foreground = increase_bkg(foreground)

    if resized is False and random.random() > (1 - (stage * 0.2)):
        foreground = rotate_char(foreground)

    if random.random() > (1 - (stage * 0.2)):
        foreground = cv2.blur(foreground, (3, 3))

    foreground = change_char_color(foreground)
    # imshow('change_char_color', foreground)

    foreground = crop_without_bkg(foreground)

    h, w = foreground.shape[:2]

    if h > max_height:
        k = max_height/h
        nw = int(w * k)
        foreground = cv2.resize(foreground, (nw, max_height))

    return foreground


def preprocess_plate(plate_image, stage):
    r = random.random()
    if r < (stage * 0.2):
        bgr = cv2.cvtColor(plate_image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        brown_lo = np.array([0, 0, 0])
        brown_hi = np.array([179, 50, 255])

        # Mask image to only select browns
        mask = cv2.inRange(hsv, brown_lo, brown_hi)

        # Change image to red where we found brown
        new_color = [random.randint(50, 255) if i < 3 else 255 for i in range(4)]
        plate_image[mask > 0] = new_color
    elif r < (stage * 0.3):
        bgr = cv2.cvtColor(plate_image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        brown_lo = np.array([0, 0, 0])
        brown_hi = np.array([179, 50, 255])

        # Mask image to only select browns
        mask = cv2.inRange(hsv, brown_lo, brown_hi)
        h, w, c = plate_image.shape
        bkg = (np.random.rand(random.randint(2, h), random.randint(2, w), 4)*255).astype('float32')
        # print(bkg.shape,(w, h))
        bkg[:, :, 3] = 255
        bkg = cv2.resize(bkg, (w, h), interpolation=cv2.INTER_LINEAR).astype('uint8')
        # print(plate_image.shape, plate_image.dtype)
        # print(bkg.shape, bkg.dtype)
        plate_image = cv2.bitwise_and(plate_image, bkg, mask=mask)

    return plate_image


def generate_plate(number, char_images, bkg_image, stage):
    last_chars_dx = 40  # place of EU flag

    generated_plate = bkg_image.copy()
    generated_plate = preprocess_plate(generated_plate, stage)

    height_plate = generated_plate.shape[0]
    max_height_char = int(height_plate * 0.9)

    threads = []
    foregrounds = []

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for n in number:
    #         future = executor.submit(preprocess_char_image, max_height_char, char_images.get(n))
    #         threads.append(future)
    #     for thread in threads:
    #         foregrounds.append(thread.result())

    for n in number:
        foregrounds.append(preprocess_char_image(max_height_char, char_images.get(n), stage))


    # print(foregrounds)
    for i, (n, foreground) in enumerate(zip(number, foregrounds)):
        if foreground is None:
            last_chars_dx += random.randint(10, 40)
            continue
        # foreground = char_images[n]

        # foreground = preprocess_char_image(foreground, max_height=max_height_char)
        h, w = foreground.shape[:2]
        y1 = int((height_plate - h) * (random.random() * 0.6 + 0.2))

        y2 = y1 + h
        x1 = i * random.randint(2, 6) + last_chars_dx
        x2 = x1 + w
        last_chars_dx = x2

        if x2 >= generated_plate.shape[1]:
            number = number[:i]
            break

        background = generated_plate[y1:y2, x1:x2]

        # assert foreground.shape == background.shape, \
        #     f'foreground: {foreground.shape}; background: {background.shape}\nx1 {x1} x2 {x2} w {w}'
        # imshow('', foreground)
        # normalize alpha channels from 0-255 to 0-1
        alpha_background = background[:, :, 3] / 255.0
        alpha_foreground = foreground[:, :, 3] / 255.0
        # set adjusted colors
        try:
            for color in range(0, 3):
                background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
                                          alpha_background * background[:, :, color] * (1 - alpha_foreground)
        except Exception as e:
            print(f'exception {e} \n{foreground.shape} {background.shape} {bkg_image.shape}'
                  f'\n{x1} {x2} {x2-x1} {w}'
                  f'\n{y1} {y2} {y2-y1} {h}\n')
            return None, ''

        # set adjusted alpha and denormalize back to 0-255
        background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

        generated_plate[y1:y2, x1:x2] = background
    generated_plate = cv2.cvtColor(generated_plate, cv2.COLOR_BGRA2BGR)
    return generated_plate, number


def get_random_data(alphabet, plate_bkgs):
    len_plate_number = random.randint(4, 8)
    count_space = random.choices([0, 1, 2], [0.5, 0.25, 0.25])[0]
    number = ' '*count_space + ''.join([random.choice(alphabet) for _ in range(len_plate_number)])
    bkg_plate = random.choice(plate_bkgs)
    return bkg_plate, number


if __name__ == '__main__':
    char_image_dict = get_char_images()
    plate_bkgs = get_bkg_plate_images()
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZaou0123456789 _'


    stage = 2

    if stage == 0:
        P = 0.1
    elif stage == 1:
        P = 0.2
    elif stage == 2:
        P = 0.3
    else:
        P = 0.4
    augmentations = []
    augmentations.append(A.PiecewiseAffine(scale=(0.01, 0.02), p=P))
    augmentations.append(A.MotionBlur(blur_limit=15, p=P+0.2))
    augmentations.append(A.GlassBlur(max_delta=2, p=P))
    augmentations.append(A.RandomBrightness(limit=(-0.6, 0.2), p=1))
    augmentations.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, src_radius=100, p=P))
    augmentations.append(A.RandomFog(p=P))
    augmentations.append(A.RandomSnow(p=P))
    augmentations.append(A.JpegCompression(quality_lower=1, quality_upper=100, p=P+0.2))
    augmentations.append(A.ImageCompression(quality_lower=1, quality_upper=100, p=P-0.1))
    augmentations.append(A.GaussNoise(var_limit=(50, 100), p=P))
    augmentations.append(A.OpticalDistortion(distort_limit=0.1, shift_limit=0.5, p=P))
    augmentations.append(A.Perspective(scale=0.05, p=P))
    augmentations.append(A.Rotate(limit=5, p=P))
    augmentations.append(A.Affine(rotate=1, shear={"x": (-10, 10), "y": (-5, 5)}, p=P))

    for _ in range(20):
        bkg_plate, number = get_random_data(alphabet, plate_bkgs)
        print(number)
        t1 = time.time()
        generated_plate, number = generate_plate(number, char_image_dict, bkg_plate, stage)
        t2 = time.time()
        generated_plate = {'image': generated_plate}

        random.shuffle(augmentations)

        for aug in augmentations:
            generated_plate = aug(**generated_plate)
        t3 = time.time()
        print(f'generate {t2 - t1}, augment {t3 - t2}')
        # print(generated_plate)
        # plt.imshow(generated_plate)
        cv2.imshow(number, generated_plate['image'])
        cv2.waitKey(0)
