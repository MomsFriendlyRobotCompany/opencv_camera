##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from colorama import Fore
from dataclasses import dataclass
import numpy as np

@dataclass
class Camera:
    """
    Nice class for holding stereo camera info
    """
    K: np.ndarray # left camera matrix
    d: np.ndarray # distortion coefficients
    shape: np.ndarray # image size that was calibrated

    def __str__(self):
        ms = lambda m: "    {}".format(str(m).replace('\n','\n    '))

        s = f'{Fore.BLUE}Camera[{self.shape}]----------------------{Fore.RESET}\n'
        s += f'  focalLength(x,y): {self.K[0,0]:.1f} {self.K[1,1]:.1f} px \n'
        s += f'  principlePoint(x,y): {self.K[0,2]:.1f} {self.K[1,2]:.1f} px\n'
        s += f'  distortionCoeffs: {self.d}\n'
        return s
