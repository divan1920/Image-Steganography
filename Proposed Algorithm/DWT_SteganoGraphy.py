# Libraries
import pathlib
import time
from anyio import open_file
from black import out
import cv2
import numpy as np
import pywt
import skimage.measure
import math
import random
from random import randint, seed
from custom_Exceptions import *
from sewar.full_ref import ssim, uqi, psnr

# region
"""##Skin Detection"""


def skin_detection(img):
    # converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    #removing the noise
    HSV_mask = cv2.morphologyEx(
        HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for YCbCr color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    #removing the noise
    YCrCb_mask = cv2.morphologyEx(
        YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    # removing noise from image
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(
        global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    # checking if there are skin pixels in image or not
    unique, counts = np.unique(global_mask, return_counts=True)
    if(unique[-1] != 255):
        raise SkinNotDetected(
            "Oops!!! Skin can not be Detected in given Image")
    return global_mask


"""##Hole Filling"""


def FillHole(mask):
    #bigger the image more time it will take 
    # it is morphological operation(dilation) in which white pixels grows 
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(
            drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
    out = sum(contour_list)
    #if out is same type as int then there is no white pixel in the image
    if type(out) == type(1):
        if(out == 0):
            raise SkinNotDetected(
                "Oops!!! Hole Filling is not Possible in this Image")
    return out


"""##Largest Connected Component"""


def largest_connected_component_coordinates(image):
    image = image.astype("uint8")
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    # states has sizes of all connected componets there is in the image and 
    # we will take it as non-increasing array
    sizes = stats[:, -1]
    # find component of size >3000
    max_label = 1
    max_size = 3000  # find component of size >3000
    # 0th component is the background component so we cannot take that
    # as it has only black pixels
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    if(max_size == 3000):
        raise LargestComponentNotFound(
            "Oops!!! Skin Component larger than 3000 can't be Found")
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    y_nonzero, x_nonzero = np.nonzero(img2)
    return (y_nonzero, x_nonzero)


"""##Crop ROI"""


def crop(y_nonzero, x_nonzero, image):
    return image[
        np.min(y_nonzero): np.max(y_nonzero), np.min(x_nonzero): np.max(x_nonzero)
    ]


"""##Get sub-bands"""


def get_subbands(ROI):
    coeffs = pywt.dwt2(ROI, "haar")
    return coeffs


"""##Get maxEntropy sub-band"""


def get_max_entropy_subband(coeffs):
    cA, (cH, cV, cD) = coeffs

    Dict = {
        "cH": skimage.measure.shannon_entropy(cH),
        "cV": skimage.measure.shannon_entropy(cV),
        "cD": skimage.measure.shannon_entropy(cD),
    }
    # find Maximum from Dictionary
    Keymax = max(zip(Dict.values(), Dict.keys()))[1]
    if Keymax == "cH":
        maxEntropy_Band = cH
    elif Keymax == "cV":
        maxEntropy_Band = cV
    elif Keymax == "cD":
        maxEntropy_Band = cD

    return maxEntropy_Band


"""##Generate binary steam from string"""


def str_to_binary(string):
    string = str(string)
    binary = "".join(format(i, "08b")
                     for i in bytearray(string, encoding="utf-8"))
    return binary


"""##Generate Coordinates"""


def generate_coordinates(ROI, length):
    m, n = ROI.shape
    m = m - 1
    n = n - 1
    random.seed(length)
    coordinates = set()
    x, y = randint(1, m), randint(1, n)
    while len(coordinates) < length:
        while (x, y) in coordinates:
            x, y = randint(1, m), randint(1, n)
        coordinates.add((x, y))
    return coordinates


"""##Hiding inside Max Entropy Band"""


def hide_in_maxband(coordinates, maxEntropy_Band, binary):
    modified_maxEntropy_Band = np.copy(maxEntropy_Band)
    i = 0
    for x, y in coordinates:

        fract_pixel, int_pixel = math.modf(maxEntropy_Band[x, y])
        bin_pixel = bin(int(int_pixel))
        new_bin_pixel = list(bin_pixel)
        new_bin_pixel[len(bin_pixel) - 1] = binary[i]
        bin_pixel = "".join(new_bin_pixel)
        int_pixel = int(bin_pixel, 2)
        modified_maxEntropy_Band[x, y] = int_pixel + fract_pixel
        i = i + 1
    return modified_maxEntropy_Band


"""##Get ROI back by Applying IDWT"""


def get_modified_ROI(coeffs, modified_max_entropy_band, maxEntropy_Band):
    cA, (cH, cV, cD) = coeffs
    comparison = maxEntropy_Band == cH
    equal_arrays_cH = comparison.all()
    comparison = maxEntropy_Band == cV
    equal_arrays_cV = comparison.all()
    comparison = maxEntropy_Band == cD
    equal_arrays_cD = comparison.all()

    if equal_arrays_cH:
        modified_cH = modified_max_entropy_band
        modified_coeffs = cA, (modified_cH, cV, cD)
    elif equal_arrays_cV:
        modified_cV = modified_max_entropy_band
        modified_coeffs = cA, (cH, modified_cV, cD)
    elif equal_arrays_cD:
        modified_cD = modified_max_entropy_band
        modified_coeffs = cA, (cH, cV, modified_cD)

    # Applying Invers IDWT (Embedding Side)
    idwt_ROI = pywt.idwt2(modified_coeffs, "haar")
    return idwt_ROI


"""##Seperating Intger and Fractional part of the modified ROI"""


def seperate_int_fract(idwt_ROI):
    rows, cols = idwt_ROI.shape
    int_ROI = np.empty((rows, cols), np.uint8)
    fract_ROI = np.empty((rows, cols), np.float64)

    for x in range(0, rows):
        for y in range(0, cols):
            fract_pixel, int_pixel = math.modf(idwt_ROI[x, y])
            if int_pixel >= 0:
                int_ROI[x, y] = int_pixel
                fract_ROI[x, y] = fract_pixel
            else:
                int_ROI[x, y] = abs(int_pixel)
                fract_ROI[x, y] = -fract_pixel
    return int_ROI, fract_ROI


"""##Merging modified ROI back to binary image"""


def merge_roi(binary_image, int_ROI, y_nonzero, x_nonzero):
    x = np.min(x_nonzero)
    y = np.min(y_nonzero)
    binary_image[y: y + int_ROI.shape[0], x: x + int_ROI.shape[1]] = int_ROI
    return binary_image


"""##Encoding function for hiding binary image into R-plane"""


def hide_in_R(image_to_hide, image_to_hide_in):
    width, height = image_to_hide_in.shape

    for y in range(height):
        for x in range(width):
            value = image_to_hide[x, y]
            pixel = image_to_hide_in[x, y]
            bin_pixel = bin(int(pixel))
            new_bin_pixel = list(bin_pixel)
            if value == 0:
                new_bin_pixel[len(bin_pixel) - 2] = "0"
                new_bin_pixel[len(bin_pixel) - 1] = "0"
            elif value == 1:
                new_bin_pixel[len(bin_pixel) - 2] = "0"
                new_bin_pixel[len(bin_pixel) - 1] = "1"
            elif value == 254:
                new_bin_pixel[len(bin_pixel) - 2] = "1"
                new_bin_pixel[len(bin_pixel) - 1] = "0"
            elif value == 255:
                new_bin_pixel[len(bin_pixel) - 2] = "1"
                new_bin_pixel[len(bin_pixel) - 1] = "1"

            bin_pixel = "".join(new_bin_pixel)
            int_pixel = int(bin_pixel, 2)
            image_to_hide_in[x, y] = int_pixel

    # return an Image object from the above data.
    return image_to_hide_in


# This code retrives image from an image using 2-LSB


def get_binary_from_R(combine_img):
    
    width, height = combine_img.shape
    hidden_img = np.zeros_like(combine_img, dtype="uint8")
    for y in range(height):
        for x in range(width):
            pixel = combine_img[x, y]
            bin_pixel = bin(int(pixel))
            new_bin_pixel = list(bin_pixel)
            if (
                new_bin_pixel[len(bin_pixel) - 2] == "0"
                and new_bin_pixel[len(bin_pixel) - 1] == "0"
            ):
                value = 0
            elif (
                new_bin_pixel[len(bin_pixel) - 2] == "0"
                and new_bin_pixel[len(bin_pixel) - 1] == "1"
            ):
                value = 1
            elif (
                new_bin_pixel[len(bin_pixel) - 2] == "1"
                and new_bin_pixel[len(bin_pixel) - 1] == "0"
            ):
                value = 254
            elif (
                new_bin_pixel[len(bin_pixel) - 2] == "1"
                and new_bin_pixel[len(bin_pixel) - 1] == "1"
            ):
                value = 255

            hidden_img[x, y] = value
    return hidden_img


"""##combine fact and int ROI"""


def combine_roi(ROI, fract):
    width, height = ROI.shape
    new_roi = np.zeros_like(ROI, dtype="float64")

    for y in range(height):
        for x in range(width):
            fractd = fract[x, y]
            if fractd < 0:
                new_roi[x, y] = -(ROI[x, y] + abs(fractd))
            else:
                new_roi[x, y] = ROI[x, y] + fractd
    return new_roi


"""##Get Message"""


def get_message(maxEntropy_Band, coordinates):
    message = []
    for x, y in coordinates:
        pixel = maxEntropy_Band[x, y]
        fract_pixel, int_pixel = math.modf(maxEntropy_Band[x, y])
        bin_pixel = bin(int(int_pixel))
        message.append(bin_pixel[len(bin_pixel) - 1])
    string_message = ""
    for x in message:
        string_message += x
    ascii_string = "".join(
        chr(int(string_message[i * 8: i * 8 + 8], 2))
        for i in range(len(string_message) // 8)
    )
    return ascii_string


"""##Check Quility"""


def get_psnr(input_Path_1, input_Path_2):
    image1 = cv2.imread(input_Path_1)
    image2 = cv2.imread(input_Path_2)
    return psnr(image1, image2)


def get_ssim(input_Path_1, input_Path_2):
    image1 = cv2.imread(input_Path_1)
    image2 = cv2.imread(input_Path_2)
    return ssim(image1, image2)[0]


def get_uqi(input_Path_1, input_Path_2):
    image1 = cv2.imread(input_Path_1)
    image2 = cv2.imread(input_Path_2)
    return uqi(image1, image2)


def check_capasity(roi, lenght):
    x, y = roi.shape
    max_Characters = math.floor((x*y)/8)
    if(lenght > (x*y)):
        raise NotEnoughCapasity(
            "Oops!!! Failed to Embedd Message into given Image \n\n Max Capasity is of '{}' Characters".format(max_Characters))

# endregion


def encode(input_Path, text, output_Path, output_FractionalPath, progressBar=None):
    image = cv2.imread(input_Path)
    binary_image = skin_detection(image)
    hole_filled_image = FillHole(binary_image)
    y_nonzero, x_nonzero = largest_connected_component_coordinates(
        hole_filled_image)
    ROI = crop(y_nonzero, x_nonzero, binary_image)
    coeffs = get_subbands(ROI)
    max_entropy_subband = get_max_entropy_subband(coeffs)
    binary = str_to_binary(text)
    # check if we can accomodate msg inside this segmant
    check_capasity(max_entropy_subband, len(binary))
    coordinates = generate_coordinates(max_entropy_subband, len(binary))
    modified_max_entropy_band = hide_in_maxband(
        coordinates, max_entropy_subband, binary)
    modified_roi = get_modified_ROI(
        coeffs, modified_max_entropy_band, max_entropy_subband)
    int_roi, fract_roi = seperate_int_fract(modified_roi)
    modified_binary_image = merge_roi(
        binary_image, int_roi, y_nonzero, x_nonzero)
    B, G, R = cv2.split(image)
    modified_r = hide_in_R(modified_binary_image, R)
    stego_image = cv2.merge([B, G, modified_r])
    seed_value = len(binary)

    # for Duplicate Files
    if pathlib.Path(output_Path).suffix == '.png':
        wrt1 = cv2.imwrite(output_Path, stego_image)
    else:
        wrt1 = cv2.imwrite(output_Path + '.png', stego_image)
    if pathlib.Path(output_FractionalPath).suffix == '.npy':
        wrt2 = np.save(output_FractionalPath, fract_roi)
    else:
        wrt2 = np.save(output_FractionalPath + '.npy', fract_roi)

    if not wrt1:
        raise FileError("Failed to write image '{}'".format(output_Path))
    else:
        # setting for loop to set value of progress bar
        for i in range(101):

            # slowing down the loop
            time.sleep(0.007)

            # setting value to progress bar
            progressBar.setValue(i)

    return seed_value


def decode(input_Path, input_FractionalPath, seed, progressBar=None):

    if(int(seed) % 8 != 0):
        raise SeedNotValid("Given seed '{}' is not Valid".format(
            seed))
    stego_image = cv2.imread(input_Path)
    fract_roi = np.load(input_FractionalPath)
    B, G, R = cv2.split(stego_image)
    binary_image = get_binary_from_R(R)
    hole_filled_image = FillHole(binary_image)
    y_nonzero, x_nonzero = largest_connected_component_coordinates(
        hole_filled_image)
    ROI = crop(y_nonzero, x_nonzero, binary_image)
    modified_ROI = combine_roi(ROI, fract_roi)
    coeffs = get_subbands(modified_ROI)
    max_entropy_subband = get_max_entropy_subband(coeffs)
    coordinates = generate_coordinates(max_entropy_subband, int(seed))
    text = get_message(max_entropy_subband, coordinates)

    for i in range(101):

        # slowing down the loop
        time.sleep(0.007)

        # setting value to progress bar
        progressBar.setValue(i)

    return text