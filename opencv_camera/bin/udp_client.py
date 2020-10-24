#!/usr/bin/env python3
# -*- coding: utf-8 -*
##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################

import socket
import cv2
import numpy as np
import struct
from threading import Thread, Lock
import argparse

class ImageGrabber(Thread):
    def __init__(self, host, port):
        Thread.__init__(self)
        self.lock = Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (host, port)
        self.sock.settimeout(0.05)
        self.array = None
        self.running = True

    def stopServer(self):
        self.sock.sendto(struct.pack('<L',0), self.server_address)

    def imageGrabber(self):
        if self.array is not None:
            self.lock.acquire()
            array = self.array
            self.lock.release()
            return array

    def run(self):
        while self.running:
            try:
                self.sock.sendto(struct.pack('<L',1), self.server_address)
                try:
                    data_len_packed, server = self.sock.recvfrom(struct.calcsize('<L'))
                except socket.timeout:
                    continue
                data_len = struct.unpack('<L',data_len_packed)[0]
                # if data_len < 65507:
                print(f"byte recv: {data_len}", end="\r")
                try:
                    data, server = self.sock.recvfrom(data_len)
                except socket.timeout:
                    continue
                except Exception:
                    continue
                if not len(data) == data_len:
                    print ("There was a image packet loss...")
                    continue
                if data == 404:
                    continue

                self.lock.acquire()
                self.array = np.frombuffer(data, dtype=np.dtype('uint8'))
                self.lock.release()

                # print(f"{self.array.shape}")

            except:
                self.running = False
                self.lock.acquire()
                self.image = None
                self.lock.release()

def handle_args():
    # parser = argparse.ArgumentParser(version=VERSION, description='A simple \
    parser = argparse.ArgumentParser(description=f'A simple \
    program to display images from a camera sent over the network \
    as a UDP message.')

    # parser.add_argument('-c', '--camera', help='which camera to use, default is 0', default=0)
    parser.add_argument('host', help='host ip address', default=None)
    # parser.add_argument('-q', '--quality', help='jpeg quality percentage, default is 80', default=80)
    parser.add_argument('-p','--port', help='port, default is 9050', default=9050)
    # parser.add_argument('-s', '--size', type=int, help='size of image capture (480=(640x480), 720=(1280x720)), default 240')
    # parser.add_argument('-v', '--version', action='store_true', help='returns version number')

    return vars(parser.parse_args())

def main():
    args = handle_args()
    host = args["host"]
    port = args["port"]

    image = ImageGrabber(host, port)
    image.daemon = True
    image.start()

    try:
        while image.running:
            array = image.imageGrabber()
            if array is None:
                continue
            img = cv2.imdecode(array, 1)
            try:
                cv2.imshow(f"{img.shape}", img)
            except Exception:
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("ctrl-C ...")
    except:
        pass
    finally:
        image.stopServer()
        image.sock.close()


if __name__ == '__main__':
    main()
