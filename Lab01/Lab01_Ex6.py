from picamera import PiCamera
import picamera.exc
import time as t
from time import sleep
import argparse
import os

def main(width, height, framerate, n_pics, output, format, sleep_time, rotation):

    """
    Script for taking pictures with the raspberry pi

    The camera preview only works when a monitor is connected to your Raspberry Pi.
    If you are using remote access (such as SSH or VNC),
    you won’t’ see the camera preview.
    """

    if sleep_time < 2:
        print("It’s important to sleep for at least two seconds before \
            capturing an image, because this gives the camera’s sensor \
                time to sense the light levels")

    if width > 2592 or height > 1944 or width < 64 or height < 64:
        print("Max width x heigh= 2592x1944")
        print("Min with x heigh: 64x64")
        raise picamera.exc.PiCameraValueError()

    folder = "/home/pi/WORK_DIR/ML4IOT/Lab01/images/"

    camera = PiCamera()
    camera.rotation = rotation
    camera.resolution = (width, height)
    camera.framerate = framerate

    camera.start_preview()
    # For transparency of the preview:
    #camera.start_preview(alpha=200) 

    # It’s important to sleep for at least two seconds before
    # capturing an image, because this gives the camera’s sensor
    # time to sense the light levels
    for i in range(n_pics):
        #camera.annotate_text = "Hello world!" + str(i)
        #camera.awb_mode = 'sunlight'
        sleep(sleep_time)

        filename = folder + output + str(i) + format
        #print(filename)

        if i == 0:
            start_picture = t.time()
            camera.capture(filename)
            end_picture = t.time()
            print(f"Time for taking and storing a picture: {end_picture-start_picture :.3f}")
            print(f"Size of an image is circa: {int(os.path.getsize(filename)/1048576)} MegaBytes")
        
        camera.capture(filename)

    camera.stop_preview()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default='1920', help='Widht of the image in pixels')
    parser.add_argument('--height', type=int, default='1080', help='Height of the image in pixels')
    parser.add_argument('--framerate', type=int, default=15, help='Framerate for the camera preview in this case')
    parser.add_argument('--n_pics', type=int, default=1, help='Number of pictures')
    parser.add_argument('--output', type=str, default='image', help='Name of the image')
    parser.add_argument('--format', type=str, default='.jpg', help='Image format')
    parser.add_argument('--sleep', type=int, default=2, help='Seconds between 2 images (at least 2 seconds recommended)')
    parser.add_argument('--rotation', type=int, default=180, help='Rotation in degrees of the camera, default is 180)')

    

    args, _ = parser.parse_known_args()

    width = args.width
    height = args.height
    framerate = args.framerate
    n_pics = args.n_pics
    output = args.output
    format = args.format
    sleep_time = args.sleep
    rotation = args.rotation

    main(width, height, framerate, n_pics, output, format, sleep_time, rotation)
