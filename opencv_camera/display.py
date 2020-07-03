# import numpy as np
import cv2
# from glob import glob
# import yaml
# import json
import attr
# from enum import Enum
import time
# from collections import namedtuple


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
