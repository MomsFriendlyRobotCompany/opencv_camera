##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import attr
import cv2             # OpenCV camera
import time            # sleep
import platform        # determine linux or darwin (OSX)
import os              # check for travis.ci environment


class VideoError(Exception):
    pass


@attr.s(slots=True)
class SaveVideo(object):
    """
    Simple class to save frames to video (mp4v)

    macOS: avc1
    windows: h264?
    """

    writer = attr.ib(default=None)
    encoder = attr.ib(default=None)

    def open(self, filename, width, height, fps=30, fourcc=None):
        self.close() # close if another is already open

        # pick a good encoder for the current OS
        if fourcc is None:
            sys = platform.system().lower()
            if sys in ['darwin']:
                fourcc = 'avc1'
            else:
                fourcc = 'mjpg'

        try:
            mpg4 = cv2.VideoWriter_fourcc(*fourcc)
        except Exception as err:
            print(err)
            print(f'Please select another encoder, {fourcc} failed')
            raise

        self.writer = cv2.VideoWriter()
        self.writer.open(filename, mpg4, fps, (width,height))

    def __del__(self):
        self.close()

    def write(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        self.writer.write(image)

    def write_list(self, frames, fname='out.mp4', fps=30):
        shape = frames[0].shape

        frame_height, frame_width = shape[:2]

        # video writer doesn't like grayscale images, have
        # to convert to RGB
        if len(shape) == 2:
            grayscale = True
        else:
            grayscale = False

        # pick a good encoder for the current OS
        sys = platform.system().lower()
        if sys in ['darwin']:
            fourcc = 'avc1'
        else:
            fourcc = 'mjpg'

        print('>> Saving {} {}x{} images to {}'.format(len(frames), shape[1], shape[0], fname))
        print('>> using {} on {}'.format(fourcc, sys))

        # create the video writer and write all frames to the file
        out = cv2.VideoWriter(
            fname,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (frame_width,frame_height))

        for frame in frames:
            # convert if necessary to RGB
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            out.write(frame)

        out.release()
        print('>> wrote {:.1f} MB'.format(os.path.getsize(fname)/(1E6)))

    def close(self):
        if self.writer:
            self.writer.release()
