##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
import attr
import time


@attr.s(slots=True)
class FlipBook:
    imgs = attr.ib()
    auto = attr.ib(default=True)

    def run(self):
        print('---------------------------------')
        print(' ESC/q to quit')
        print(' spacebar to pause/continue')
        print('---------------------------------')

        s = self.imgs[0].shape
        if self.auto:
            delay = 500
        else:
            delay = 0

        for img in self.imgs:
            cv2.imshow(f"{s}", img)

            ch = cv2.waitKey(delay)
            if ch in [27, ord('q')]:
                break
            elif ch == ord(' '):
                delay = 0 if delay > 0 else 500

        cv2.destroyAllWindows()
