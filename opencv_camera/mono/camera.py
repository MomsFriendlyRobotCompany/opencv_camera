##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from colorama import Fore
from dataclasses import dataclass
import numpy as np
from ..undistort import UnDistort
from pathlib import Path
import yaml


@dataclass
class Camera:
    """
    Nice class for holding stereo camera info

    K: camera matrix
    d: distortion coefficients
    h: image height/rows
    w: image width/columns
    """
    K: np.ndarray # left camera matrix
    d: np.ndarray # distortion coefficients
    # shape: np.ndarray # image size that was calibrated
    h: int # height/rows
    w: int # width/columns

    def __str__(self):
        ms = lambda m: "    {}".format(str(m).replace('\n','\n    '))

        s = f'{Fore.BLUE}Camera[{self.h},{self.w}]----------------------{Fore.RESET}\n'
        s += f'  focalLength(x,y): {self.K[0,0]:.1f} {self.K[1,1]:.1f} px \n'
        s += f'  principlePoint(x,y): {self.K[0,2]:.1f} {self.K[1,2]:.1f} px\n'
        s += f'  distortionCoeffs: {self.d}\n'
        return s

    def getUndistortion(self):
        return UnDistort(
            self.K,
            self.d,
            self.h,
            self.w
        )

    @classmethod
    def from_yaml(cls, file):
        if not isinstance(file, Path):
            p = Path(file)
        p = p.expanduser().resolve()

        with p.open(mode="r") as fd:
            info = yaml.safe_load(fd)

        args = []
        for key in ["K","d","h","w"]:
            args.append(np.array(info[key]))

        c = cls(*args)

        return c

    def to_yaml(self, filename):
        camera_parameters = {
            "K": self.K.tolist(),
            "d": self.d.tolist(),
            "h": self.h,
            "w": self.w,
        }

        p = Path(filename).expanduser().resolve()
        with p.open("w") as fd:
            yaml.safe_dump(camera_parameters, fd)
