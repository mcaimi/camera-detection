#!/usr/bin/env python

try:
    import cv2 as cv
    import numpy as np

    # local libs
    from .imageio import splitRGBChannels, splitHSVChannels
except Exception as e:
    print(f"Caught exception: {e}")


# compute rgb histogram
def calcRGBHistogram(image: np.ndarray, buckets: int) -> list[np.ndarray]:
    # define bucket sizes
    r_size: int = buckets
    g_size: int = buckets
    b_size: int = buckets

    # split image into component channels
    r, g, b = splitRGBChannels(image)

    # calculate histogram
    r_hist: np.ndarray = cv.calcHist([r], channels=[0], mask=None, histSize=[r_size], ranges=[0, r_size])
    g_hist: np.ndarray = cv.calcHist([g], channels=[0], mask=None, histSize=[g_size], ranges=[0, g_size])
    b_hist: np.ndarray = cv.calcHist([b], channels=[0], mask=None, histSize=[b_size], ranges=[0, b_size])

    # return histogram
    return [r_hist, g_hist, b_hist]


# compute HSV histogram
def calcHSVHistogram(image: np.ndarray, buckets: int) -> list[np.ndarray]:
    # define bucket sizes
    h_size: int = buckets
    s_size: int = buckets
    v_size: int = buckets

    # split image into component channels
    h, s, v = splitHSVChannels(image)

    # calculate histogram
    h_hist: np.ndarray = cv.calcHist([h], channels=[0], mask=None, histSize=[h_size], ranges=[0, h_size])
    s_hist: np.ndarray = cv.calcHist([s], channels=[0], mask=None, histSize=[s_size], ranges=[0, s_size])
    v_hist: np.ndarray = cv.calcHist([v], channels=[0], mask=None, histSize=[v_size], ranges=[0, v_size])

    # return histogram
    return [h_hist, s_hist, v_hist]


# display bar histogram as image
def barHistogram(channel_histogram: np.ndarray, 
                 bins: int, histogram_dimensions: tuple = (512, 384),
                 color: tuple = (0, 0, 0), thickness: int = 4) -> np.ndarray:
    # histogram dimensions
    graph_width: int = histogram_dimensions[0]
    graph_height: int = histogram_dimensions[1]

    bin_width: int = int(graph_width // bins)
    data_increment: float = (graph_height / channel_histogram.max())

    # empty histogram image
    chan_img: np.ndarray = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

    # draw histogram bars
    for x in range(bins):
        chan_img = cv.rectangle(chan_img, 
                                (x*bin_width, int(channel_histogram[x][0] * data_increment)),
                                (x*bin_width+bin_width, 0),
                                color=color, thickness=thickness
                               )
    # return histogram image
    return chan_img


# line hostogram
def lineHistogram(channel_histogram: np.ndarray, bins: int, histogram_dimensions: tuple = (384, 512), color: tuple = (0, 0, 0)) -> np.ndarray:
    # get dims
    graph_width: int = histogram_dimensions[1]
    graph_height: int = histogram_dimensions[0]
    data_increment: float = (graph_height // channel_histogram.max())
    line_thickness: int = int(graph_width // bins)

    # empty histogram image
    chan_img: np.ndarray = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

    for i, h in enumerate(channel_histogram):
        cv.line(chan_img, (i * line_thickness, int(h * data_increment)), (i * line_thickness, (graph_height - int(h))), color=color, thickness=line_thickness)

    # return histogram image
    return chan_img


# combine histograms in a single image with blending
def combineHistograms(hists: list[np.ndarray], blending: float) -> np.ndarray:
    # prepare output array
    background: np.ndarray = np.zeros_like(hists[0], dtype=np.uint8)

    # compute masks
    grayscale_hists: list[np.ndarray] = [cv.cvtColor(hist, cv.COLOR_RGB2GRAY) for hist in hists]
    masks: list[np.ndarray] = [cv.threshold(hist, 10, 255, cv.THRESH_BINARY)[1] for hist in grayscale_hists]
    inverse_masks: list[np.ndarray] = [cv.bitwise_not(m) for m in masks]

    # prepare merge zones
    for h, m, im in zip(hists, masks, inverse_masks):
        excluded: np.ndarray = cv.bitwise_and(background, background, mask=im)
        included: np.ndarray = cv.bitwise_and(h, h, mask=m)
        hist_combined = cv.addWeighted(excluded, 1.0, included, blending, gamma=1)
        background = hist_combined.copy()

    # return histogram image
    return hist_combined


# scale histogram to fit on overlay
def scaleArray(imageToResize: np.ndarray,
               reference: np.ndarray,
               xscale: float, yscale: float,
               interpolation=cv.INTER_AREA) -> np.ndarray:
    # resize
    resized_image: np.ndarray = cv.resize(imageToResize, None,
                                          fx=xscale, fy=yscale,
                                          interpolation=interpolation)

    # return image
    return resized_image


# overlay histogram on image
def overlayHistogram(histogram: np.ndarray, frame: np.ndarray) -> np.ndarray:
    # get boundaries
    img_rows, img_cols, img_chan = frame.shape
    hist_rows, hist_cols, hist_chan = histogram.shape

    # calculate overlay regions
    roi: np.ndarray = frame[(img_rows - hist_rows):, (img_cols - hist_cols):]
    roi = cv.flip(histogram, 0).copy()
    frame[(img_rows - hist_rows):, (img_cols - hist_cols):] = roi

    # return array
    return frame
