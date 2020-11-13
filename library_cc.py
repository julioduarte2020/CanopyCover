# -*- coding: utf-8 -*-
# LIBRARY OF CaCo

'''
Canopy Cover (CaCo) V0.1

===========================================================
An objective image analysis method for estimation of canopy
attributes from digital cover photography

* author: Alessandro Alivernini <alessandro.alivernini@crea.gov.it>
* paper: https://doi.org/10.1007/s00468-018-1666-3
* git: https://github.com/alivernini/caco

===========================================================

Canopy Cover (CaCo)
Copyright 2017-2018 Council for Agricultural Research and Economics

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


# @@ MODULE IMPORT
import os
import math
import numpy as np
import pandas as pds
import rawpy

import scipy.ndimage

import imageio
import skimage
import skimage.io
import skimage.filters

from .threshold_minimum import *

#import skimage.feature
#from   skimage.measure  import label


class CacoImg():
    '''
    Assess the gap fraction for a single img
    '''

    def __init__(self, param3, input_image):  # @@
        ''' <1> read image and set the output dir  '''
        self.param3 = param3
        self.warning = []   # store output errors

        self.path = input_image
        self.read_img(input_image)


    def run(self):  # @@
        '''
        Performs all the gap fraction computations
        '''
        if self.check_warning(): return
        self.select_band()
        self.apply_threshold()
        if self.check_warning(): return
        self.get_regions()
        return self.assess_gap_fraction()

        print("CaCo img complete")



    def select_band(self):  # @@
        ''' <2> define the band (or the combination) used '''
        band = self.param3['band']

        bs_setter = {
            'grey'     : self.set_grey,
            'greeness' : self.set_greeness,
            'red'      : [self.set_rgb, 1],
            'green'    : [self.set_rgb, 2],
            'blue'     : [self.set_rgb, 3],
        }
        setter = bs_setter[band]
        try:
            setter()
        except: # rgb case
            setter[0](setter[1])

        self.selection_shape = self.selection_img.shape
        self.image = None

    def apply_threshold(self):  # @@
        ''' <3> apply threshold to selected band '''

        # SELECT THRESHOLD
        img = self.selection_img
        th_setter = {
            'otzu': skimage.filters.threshold_otsu,
            'isodata': skimage.filters.threshold_isodata,
            'minimum': [threshold_minimum, 128]
        }
        setter = th_setter[self.param3['threshold']]
        try:
            threshold = setter(self.selection_img)
        except(TypeError):  # the function has 2+ arguments
            try:
                threshold = setter[0](self.selection_img, setter[1])
            except Exception as e:
                print(e)
                threshold = skimage.filters.threshold_isodata(img)

        # APPLY THRESHOLD
        self.selection_img = img > threshold

    def get_regions(self):  # @@
        ''' <4> define regions for gap pixels '''
        self.region, self.id = scipy.ndimage.measurements.label(self.selection_img)

    def assess_gap_fraction(self):  # @@
        ''' <5> assess gap fraction '''
        # trick 1: img to vector
        region = self.region.flatten()
        shape_flatten = region.shape
        # trick 2: get the direct mapping of pixel having the the same label [see previous step]
        sort_index = np.argsort(region)
        # trick 3: get the reverse mapping for the happy ending
        reverse_sorted_index = np.argsort(sort_index)
        #--------------------------------------------------------------------------------
        # APPLY DIRECT MAPPING TO SAMPLE THE SIZE OF EACH IMG REGION
        sorted_region = region[sort_index]
        img = self.selection_img.flatten()[sort_index]
        #----------------------------------------------------------------------------
        # TWO LOOPS ARE PERFOMED:
        # 1. to define the threshold dividing normal gaps from big gaps;
        # 2. to sample the size of the gaps for the normal gaps

        big_gaps_defined  = False  # True at the end of the 1st loop
        output            = np.zeros(shape_flatten, dtype=np.uint8)  # stores the values of the output image
        gap_size_2        = []  # big and normal gaps
        normal_gap_size_2 = []
        big_gap_size_2    = []
        for unused in range(2):
            start = 0
            img_end = shape_flatten[0] - 1 # total of pixels -1
            np_sum = np.sum  # small trick for speed
            # -->> loop start x 2
            while start < img_end:
                fid = sorted_region[start]  # fid is equal to the new label value [remember get_regions?]
                end = start
                if end == img_end:
                    break
                # --------------------------------------------------------------------------------
                # end continues to move 1 step ahead while the labelis equal
                while fid == sorted_region[end] and end < img_end:
                    end += 1 # end finishes one step beyond the edge
                # --------------------------------------------------------------------------------
                if fid > 0:  # this is not vegetation
                    stat = np_sum(img[start:end])  # start and end are at the extremes of one label [end is 1 step beyond]
                    if not big_gaps_defined:  # 1st loop case
                        gap_size_2.append(stat)
                    else:  # 2nd loop case
                        if stat > big_gap_size:
                            output[start:end] = 2
                            big_gap_size_2.append(stat)
                        else:
                            output[start:end] = 1
                            normal_gap_size_2.append(stat)
                start = end # end was already on a new label value
            # <<-- loop end x 2
            if not big_gaps_defined:  # 1st loop case: compute stats
                big_gaps_defined = True
                gap_size_20      = np.array(gap_size_2)
                gap_mean         = gap_size_20.mean()
                gap_std_dev      = gap_size_20.std()
                gap_std_err      = float(gap_std_dev) / float(len(gap_size_20))**0.5
                big_gap_size     = gap_mean + gap_std_err
                gap_px           = gap_size_20.sum()
            else:
                # finish loop and complete the statistics
                pass

        # normal gap statistics
        tmp = np.array(normal_gap_size_2)
        normal_gap_px = tmp.sum()
     
        # big gap statistics
        tmp = np.array(big_gap_size_2)
        big_gap_px = tmp.sum()
     
        # trick 4: get the bunny out of the hat
        output = output[reverse_sorted_index]
        output = output.reshape(self.selection_shape)

        #TODO
        self.write_output_img(output)

        # compute the gap fraction
        extinction_coeff   = 0.5
        image_px           = output.shape[0] * output.shape[1]
        gap_fraction       = float(gap_px) / float(image_px)
        large_gap_fraction = float(big_gap_px)/float(image_px)
        foliage_cover      = float(1 - gap_px / image_px)
        crown_cover       = float(1 - large_gap_fraction)
        crown_porosity     = float(1 - foliage_cover / crown_cover)
        clumping_index     = float(1 - crown_porosity) * math.log(1 - foliage_cover) \
                             / (foliage_cover * math.log(crown_porosity))
        LAI = -crown_cover * math.log(crown_porosity) / extinction_coeff 
        
        # prepare the dictionary for the desired statistics
        data_output = {
            'gap_fraction'         : gap_fraction,
            'large_gap_fraction'   : large_gap_fraction,
            'foliage_cover'        : foliage_cover,
            'crown_cover'          : crown_cover,
            'crown_porosity'       : crown_porosity,
            'clumping_index'       : clumping_index,
            'LAI'                  : LAI
        }
        return data_output

    #--------------------------------------------------------------------------------
    ## Setter methods ##

    def set_greeness(self):  # @@
        ''' select greeness band combination'''
        self.selection_img = (
            self.image[:, :, 1]
            * 2 - (self.image[:, :, 0]
            + self.image[:, :, 2])
        )

    # https://it.mathworks.com/help/matlab/ref/rgb2gray.html
    def set_grey(self):   # @@
        ''' select grey band combination'''
        self.selection_img = (
            self.image[:, :, 0] * 0.2989 +
            self.image[:, :, 1] * 0.5870 +
            self.image[:, :, 2] * 0.1140
        ).astype(self.image.dtype)


    def set_rgb(self, band):   # @@
        ''' select one band '''
        self.selection_img = self.image[:, :, band]
    #--------------------------------------------------------------------------------
    ## Manage warning ##

    def check_warning(self):
        warning = False
        if self.warning:
            warning = True
            for wkey in self.warning:
                print(wkey)
        return warning

    #--------------------------------------------------------------------------------
    ## Input/output methods methods ##

    def write_output_img(self, gap_fraction_img):   # @@
        o_dir = self.param3['output_dir']
        if not os.path.exists(o_dir):
            os.mkdir(o_dir)
        fileName = os.path.basename(self.path)
        fileName = fileName.split('.')[0] + '.jpg'
        path = os.path.join(self.param3['output_dir'],self.param3['th_dir'], fileName)
        imageio.imsave(path, gap_fraction_img)

    def read_img(self, input_image):  # @@
        ''' read the image and set the output dir '''
        try:
            if self.param3['raw_processing'] == True:
                raw = rawpy.imread(input_image)
                self.image = raw.postprocess()
            else:  # read common image formats
                self.image = imageio.imread(input_image)
        except:  # read raw data
            print(e)
            self.error.append(['inputW'])

