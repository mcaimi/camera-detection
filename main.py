#!/usr/bin/env python

# python video capture service

import sys
try:
    import cv2
except Exception as e:
    print(f"Cannot import OpenCV. This library is mandatory, follow instructions for your distro in order to install it. {e}")
    sys.exit(-1)

try:
    from yaml import YAMLError
    from argparse import ArgumentParser
    from libs.utils import getCaptureDevice, getAcceleratorDevice, loadAndPrepareModel
    from libs.parameters import loadConfig, Parameters
    from libs.console_utils import ANSIColors
    from libs.imageio import displayFrameInfo, detectedObjects
    from libs.histogram import calcRGBHistogram, barHistogram, scaleArray, overlayHistogram, combineHistograms
except Exception as e:
    print(f"Cannot import library: {e}")
    sys.exit(-1)


# main function
def main():
    consoleIO: ANSIColors = ANSIColors()

    # parse command line
    parser = ArgumentParser(prog="AI Video Tracker", description="Detect things in a video feed")
    parser.add_argument("-c", "--config_file", action="store", required=True)
    arguments = parser.parse_args()
    try:
        configParams: Parameters = loadConfig(arguments.config_file)
    except YAMLError as e:
        consoleIO.error(f"YAML Syntax Error: {e}")
        exit(1)
    except Exception as e:
        consoleIO.error(f"Caught Exception: {e}")
        exit(1)

    consoleIO.print_success(f"Opening Capture Device: {configParams.params.captureDevice}")
    cap = getCaptureDevice(configParams.params.captureDevice)

    accelerator, dtype = getAcceleratorDevice()
    consoleIO.print_warning(f"Accelerator Configuration: DEVICE {accelerator}, DTYPE {dtype}")

    consoleIO.print_success(f"Downloading model: {configParams.huggingface.modelName}")
    model = loadAndPrepareModel(configParams).to(accelerator)

    # analyze video feed
    while (True):
        # frame capture
        r, frame = cap.read()

        # we have a frame
        if r:
            # run the object detection model on the current frame
            outputs = model(source=frame, stream=configParams.params.streamOperation)

            # display frame info
            displayFrameInfo(frame, cv2.FONT_HERSHEY_SIMPLEX, 10, 10)

            # analyze and display detected features
            frame = detectedObjects(outputs, frame)

            # compute RGB Histogram of the current frame
            rh, gh, bh = calcRGBHistogram(frame, 64)
            rg = barHistogram(rh, 64, color=(0, 0, 255))
            gg = barHistogram(gh, 64, color=(0, 255, 0))
            bg = barHistogram(bh, 64, color=(255, 0, 0))
            comb = combineHistograms([rg, gg, bg], 180)
            resized_hg = scaleArray(comb, frame, 1/2, 1/2)
            target_frame = overlayHistogram(resized_hg, frame)

            # display target frame
            cv2.imshow('Camera', target_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # unload model
    consoleIO.print_warning("Unloading model...")
    del model

    # release OpenCV Capture Device
    consoleIO.print_warning("Closing Capture Device")
    cap.release()
    cv2.destroyAllWindows()


# start execution
if __name__ == "__main__":
    main()
