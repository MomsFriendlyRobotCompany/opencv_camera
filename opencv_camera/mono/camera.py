##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import attr
from colorama import Fore


@attr.s(slots=True)
class Camera:
    """
    Nice class for holding stereo camera info
    """
    K = attr.ib() # left camera matrix
    d = attr.ib() # distortion coefficients
    shape = attr.ib() # image size that was calibrated

    def __str__(self):
        ms = lambda m: "    {}".format(str(m).replace('\n','\n    '))
        s = f'{Fore.BLUE}Camera --------------------------\n'
        s += f'  focalLength(x,y): {self.K[0,0]:.1f} {self.K[1,1]:.1f} px \n'
        s += f'  principlePoint(x,y): {self.K[0,2]:.1f} {self.K[1,2]:.1f} px\n'
        s += f'  distortionCoeffs: {self.d}\n'
