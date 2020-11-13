# -*- coding: utf-8 -*-
# =====================
# Test module of CaCo

from caco import main_cc
import os

# define input and output directories
prj_dir = 'D:/Escritorio/Corpoica/Canopy Attributes/caco_myprj'# set project root (pr)
i_dir =  'input'  # pr subdir
o_dir = 'output'  # pr subdir

# get default parameters and set input/output
param3 = main_cc.get_param3()
param3['input_dir']  = os.path.join(prj_dir, i_dir)
param3['output_dir'] = os.path.join(prj_dir, o_dir)
param3['raw_processing'] = False
#param3['band'] = 'greeness'

# run CaCo
main_cc.caco_all(param3)



