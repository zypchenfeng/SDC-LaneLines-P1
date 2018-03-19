#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# % matplotlib inline
#reading in an image
image = mpimg.imread('.\\test_images\solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    print('Pause here for now')


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    return lines


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def seperate_lines(lines):
    left = []
    right = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)  # slop of the points
            if m >= 0:
                right.append([x1, y1, x2, y2, m]) # note the image is flipped up side down
            else:
                left.append([x1, y1, x2, y2, m])
    return left, right

def slop_filter(data, cutoff, threshold = 0.08): # filter out the outliers
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4])
    return data[(data[:, 4] >= m - threshold) & (data[:, 4] <= m + threshold)]

def line_regress(data):
    data = np.array(data)
    x = np.reshape(data[:, [0, 2]], (1, len(data) * 2))[0] # extract x as single array
    y = np.reshape(data[:, [1, 3]], (1, len(data) * 2))[0] # extract y as single array
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    x = np.array(x)
    y = np.array(x*m + c)
    return x, y, m, c

def ext_lines(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    x = x2 + (x2 - x1)/line_len * length
    y = y2 + (y2 - y1)/line_len * length
    return int(x), int(y)

def image_processing(image):
    # change image to gray scale
    gray = grayscale(image)
    cpy_image = np.copy(image)

    # do the Gaussian blur first
    blur_size = 5             # kernel size for gaussian blur
    gray_cpy = np.copy(gray)  # make a copy of the gray scale image
    img_blur = gaussian_blur(gray_cpy, blur_size)

    # Canny process to find the edges
    low_threshold = 80   # low threshold for canny
    high_threshold = 160 # high threshold for canny
    edges = canny(img_blur, low_threshold, high_threshold)
    # plt.imshow(edges, cmap='Greys_r')
    # plt.show()

    # define mask
    [y_size, x_size, _] = image.shape
    vertices = np.array([[(0.1*x_size, 1*y_size), (0.47*x_size, 0.6*y_size),
                        (0.55 * x_size, 0.6 * y_size), (1*x_size, 1*y_size)]], dtype=np.int32) # define polygon for mask
    masked_edges = region_of_interest(edges, vertices)
    # plt.imshow(masked_edges, cmap='Greys_r')
    # plt.show()

    # define and do Hough transform
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = 2 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    left, right = seperate_lines(lines)

    left = slop_filter(left, cutoff=(-0.85, -0.6))
    x, y, m, c = line_regress(left)
    y_min = np.min(y)
    y_max = np.max(y)
    bot = np.array([(y_max - c) / m, y_max], dtype=int)
    top = np.array([(y_min - c) / m, y_min], dtype=int)
    left_etopx, left_etopy = ext_lines(top[0], top[1], bot[0], bot[1], 1000)
    left_ebotx, left_eboty = ext_lines(bot[0], bot[1], top[0], top[1], 1000)
    left_line = np.array([left_etopx, left_etopy, left_ebotx, left_eboty], dtype=int)

    right = slop_filter(right, cutoff=(0.45, 0.75))
    x_r, y_r, m_r, c_r = line_regress(right)
    y_r_min = np.min(y_r)
    y_r_max = np.max(y_r)
    bot_r = np.array([(y_r_max - c_r) / m_r, y_r_max], dtype=int)
    top_r = np.array([(y_r_min - c_r) / m_r, y_r_min], dtype=int)
    right_etopx, right_etopy = ext_lines(top_r[0], top_r[1], bot_r[0], bot_r[1], 300)
    right_ebotx, right_eboty = ext_lines(bot_r[0], bot_r[1], top_r[0], top_r[1], 300)
    right_line = np.array([right_etopx, right_etopy, right_ebotx, right_eboty], dtype=int)
    lines_merge = np.array([left_line, right_line])
    for x1, y1, x2, y2 in lines_merge:
        cv2.line(line_image, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=8)
    # combine both images together
    line_image = region_of_interest(line_image, vertices) # need to use the same mask to cut the extended image
    return weighted_img(line_image, cpy_image)

import os
os.listdir(".\\test_images")
for file in os.listdir(".\\test_images"):
    if file.endswith(".jpg"):
        joinpath = os.path.join(".\\test_images", file)
        imagc = mpimg.imread(joinpath)
        pst_image = image_processing(imagc)
        joinpath2 = os.path.join(".\\test_images_output", file)
        plt.imsave(joinpath2, pst_image)


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = image_processing(image)
    # plt.imshow(result)
    # plt.show()
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)