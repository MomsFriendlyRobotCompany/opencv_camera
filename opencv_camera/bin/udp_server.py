#!/usr/bin/env python3
# -*- coding: utf-8 -*
##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################

import socket
import cv2
from threading import Thread, Lock
import struct
import argparse
from colorama import Fore
from opencv_camera import __version__ as version

debug = True
host_name = socket.gethostname()

class VideoGrabber(Thread):
        def __init__(self, jpeg_quality, size=480, source=0, gray=False):
            Thread.__init__(self)
            self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            self.cap = cv2.VideoCapture(source)
            self.grayscale = gray
            # 1280x720
            # 640x480
            # 320x240
            if size == 240:
                self.cap.set(3, 320)
                self.cap.set(4, 240)
            elif size == 480:
                self.cap.set(3, 640)
                self.cap.set(4, 480)
            elif size == 720:
                self.cap.set(3, 1280)
                self.cap.set(4, 720)
            else:
                print(f"{Fore.RED}*** Invalide image size: {size} ***{Fore.RESET}")
                print("Using camera default")

            self.running = True
            self.buffer = None
            self.lock = Lock()

        def stop(self):
            self.running = False

        def get_buffer(self):

            if self.buffer is not None:
                    self.lock.acquire()
                    cpy = self.buffer.copy()
                    self.lock.release()
                    return cpy

        def run(self):
            while self.running:
                ok, img = self.cap.read()
                if not ok:
                    continue

                if self.grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # print(img.shape)
                self.lock.acquire()
                result, self.buffer = cv2.imencode('.jpg', img, self.encode_param)
                self.lock.release()

def handle_args():
    # parser = argparse.ArgumentParser(version=VERSION, description='A simple \
    parser = argparse.ArgumentParser(description=f'A simple \
    program to capture images from a camera and send them over the network \
    as a UDP message. Unfortunately, you cannot send large images. The \
    messages are limited to 65507 bytes. So the larger the image, the lower \
    jpeg quality needs to be. You can easily do 240 @ 100% or 480 @ 95% \
    or 720 @ 65%.')

    parser.add_argument('-c', '--camera', help='which camera to use, default is 0', default=0)
    parser.add_argument('-g', '--grayscale', action='store_true', help='capture grayscale images, reduces data size')
    parser.add_argument('host', help='host ip address', default=None)
    parser.add_argument('-q', '--quality', type=int, help='jpeg quality percentage, default is 80', default=80)
    parser.add_argument('-p','--port', help='port, default is 9050', default=9050)
    parser.add_argument('-s', '--size', type=int, help='size of image capture (480=(640x480), 720=(1280x720)), default 240', default=240)
    parser.add_argument('-v', '--version', action='store_true', help='returns version number')

    return vars(parser.parse_args())

def main():
    args = handle_args()

    if args["version"]:
        print(f">> udp_server version {version}")
        exit(0)

    port = args["port"]
    host = args["host"]
    jpeg_quality = args["quality"]
    size = args["size"]
    camera = args["camera"]
    gray = args["grayscale"]

    grabber = VideoGrabber(jpeg_quality, size, camera, gray)
    grabber.daemon = True
    grabber.start()

    running = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    server_address = (host, port)

    print(f'starting up on {host_name}[{host}] port {port}\n')

    sock.bind(server_address)
    try:
        while running:
            data_packed, address = sock.recvfrom(struct.calcsize('<L'))
            data = struct.unpack('<L',data_packed)[0]
            # print(data)
            if data == 1:
                buffer = grabber.get_buffer()
                if buffer is None:
                    continue
                if len(buffer) > 65507:
                    print (f"{Fore.RED}*** Image exceeds UDP message size (65507): {len(buffer)} ***{Fore.RESET}")
                    sock.sendto(struct.pack('<L',struct.calcsize('<L')), address)
                    sock.sendto(struct.pack('L',404), address) #capture error
                    continue
                sock.sendto(struct.pack('<L',len(buffer)), address)
                sock.sendto(buffer.tobytes(), address)
            elif data == 0:
                grabber.stop()
                running = False
    except KeyboardInterrupt:
        print("ctrl-C ...")
    except Exception as e:
        print(f"{Fore.RED}*** {e} ***{Fore.RESET}")

    grabber.stop()
    running = False
    print("Quitting..")
    grabber.join()
    sock.close()


if __name__ == '__main__':
    main()
