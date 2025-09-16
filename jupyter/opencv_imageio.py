# image manipulation primitives
# implemented using OpenCV & NumPY
try:
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Caught exception: {e}")


# plot image
def plot(img: np.ndarray, title: str) -> None:
    plt.imshow(img)
    plt.title(title)
    plt.show()


# show RGB histogram
def plotRGBHistogram(hist: list[np.ndarray], limits: list, color: str = "black", title:str = "Histogram") -> None:
    if type(hist) is not list:
        raise Exception("Parameter hist must be a list of ndarrays")
    # plot rgb histogram
    channels = {"r", "g", "b"}
    for i,chan in enumerate(channels):
        plt.plot(hist[i], color=chan)
        plt.xlim(limits)
        plt.xlabel("Buckets")
        plt.ylabel("RGB Value")
        plt.title(title)
    # show histogram
    plt.show()


# show HSV histogram
def plotHSVHistogram(hist: list[np.ndarray], limits: list, color: str = "black", title:str = "Histogram") -> None:
    if type(hist) is not list:
        raise Exception("Parameter hist must be a list of ndarrays")
    # plot HSV histogram
    channels = {"r", "g", "b"}
    for i,chan in enumerate(channels):
        plt.plot(hist[i], color=chan)
        plt.xlim(limits)
        plt.xlabel("Buckets")
        plt.ylabel("HSV Value")
        plt.title(title)
    # show histogram
    plt.show()


# extract color planes from image
def components(image: np.ndarray) -> list[np.ndarray]:
    h,w = image.shape[:2]

    # extract component planes from an image
    components: list = [image[:,:,c] for c in range(image.shape[2])]

    # null ndarray plane
    z: np.ndarray = np.zeros((h,w), dtype=image.dtype)

    # return data planes
    return [np.stack([components[0], z, z], axis=-1),
            np.stack([z, components[1], z], axis=-1),
            np.stack([z, z, components[2]], axis=-1)]


# split RGB channels
def splitRGBChannels(image: np.ndarray) -> list[np.ndarray]:
    # get color channels
    r, g, b = cv.split(image)

    # return channels
    return [r, g, b]


# split HSV channels
def splitHSVChannels(image: np.ndarray) -> list[np.ndarray]:
    # convert image from RGB to HSV
    image_hsv: np.ndarray = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # get channels 
    h, s, v = cv.split(image_hsv)

    # return channels
    return [h, s, v]


# compute RGB histogram
def calcRGBHistogram(image: np.ndarray, buckets: int) -> list[np.ndarray]:
    # define bucket sizes
    r_size: int = buckets
    g_size: int = buckets
    b_size: int = buckets

    # split image into component channels 
    r, g, b = splitRGBChannels(image)

    # calculate histogram
    r_hist: np.ndarray = cv.calcHist([r], channels=[0], mask=None, histSize=[r_size], ranges=[0,r_size])
    g_hist: np.ndarray = cv.calcHist([g], channels=[0], mask=None, histSize=[g_size], ranges=[0,g_size])
    b_hist: np.ndarray = cv.calcHist([b], channels=[0], mask=None, histSize=[b_size], ranges=[0,b_size])

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
    h_hist: np.ndarray = cv.calcHist([h], channels=[0], mask=None, histSize=[h_size], ranges=[0,h_size])
    s_hist: np.ndarray = cv.calcHist([s], channels=[0], mask=None, histSize=[s_size], ranges=[0,s_size])
    v_hist: np.ndarray = cv.calcHist([v], channels=[0], mask=None, histSize=[v_size], ranges=[0,v_size])

    # return histogram
    return [h_hist, s_hist, v_hist]


# display bar histogram as image
def barHistogram(channel_histogram: np.ndarray, 
                 bins: int, histogram_dimensions: tuple = (512, 384),
                 color: tuple = (0,0,0), thickness: int = 4) -> np.ndarray:
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


# combine histograms in a single image with blending
def combineHistograms(hists: list[np.ndarray], blending: float) -> np.ndarray:
    # prepare output array
    background: np.ndarray = np.zeros_like(hists[0], dtype=np.uint8)

    # compute masks
    grayscale_hists: list[np.ndarray] = [ cv.cvtColor(hist, cv.COLOR_RGB2GRAY) for hist in hists]
    masks: list[np.ndarray] = [ cv.threshold(hist, 10, 255, cv.THRESH_BINARY)[1] for hist in grayscale_hists ]
    inverse_masks: list[np.ndarray] = [ cv.bitwise_not(m) for m in masks ]

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
    # santy check
    if (reference.shape < imageToResize.shape):
        resized_image: np.ndarray = cv.resize(imageToResize, None,
                                              fx=xscale, fy=yscale,
                                              interpolation=interpolation)
        # update histogram size
        hist_rows, hist_cols, hist_chans = resized_image.shape
        print(f"Resized Histogram H: {hist_rows} W: {hist_cols} Chan: {hist_chans}")
    else:
        pass

    # return image
    return resized_image


# overlay histogram on image
def overlayHistogram(histogram: np.ndarray, frame: np.ndarray,
                     scale: tuple = (1/6, 1/6)) -> np.ndarray:
    # get boundaries
    img_rows, img_cols, img_chan = frame.shape
    hist_rows, hist_cols, hist_chan = histogram.shape

    # calculate overlay regions
    roi: np.ndarray = frame[(img_rows-hist_rows):, (img_cols-hist_cols):]
    roi = cv.flip(histogram, 0).copy()
    frame[(img_rows-hist_rows):, (img_cols-hist_cols):] = roi

    # return array
    return frame