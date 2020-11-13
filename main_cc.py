# -*- coding: utf-8 -*-
# =====================
# MAIN MODULE OF CaCo


'''
Canopy Cover (CaCo) V0.1

===========================================================
An objective image analysis method for estimation of canopy
attributes from digital cover photography

* author: Alessandro Alivernini <alessandro.alivernini@crea.gov.it>
* paper: https://doi.org/10.1007/s00468-018-1666-3
* git: https://github.com/alivernini/caco

CaCo:
    > processes every file in the input directory as a photo
    > returns an xls spreadsheet with the gap fraction of each photo
    > defines every procesing option in the PARAM3 dictionary
    > Free and Open Source software released under MIT licence

What features in the next releases?
    > graphical user interface
    > save/restore settings

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


from .library_cc import *
pds.set_option('max_columns', 30)


def get_param3():
    """
    Return the default dictionary to start CaCo software
    ====================================================
    param3 defines all the CaCo parameters
    """

    # output statistics stored in the xls spreadsheet
    keys = (  # @@
        'gap_fraction'           ,
        'large_gap_fraction'     ,
        'foliage_cover'      ,
        'crown_cover'       ,
        'crown_porosity'     ,
        'clumping_index'     ,
        'LAI'     
    )

    # available choices to define param3['band']
    band_choices = ['red', 'green', 'blue', 'greeness', 'grey']

    th_choices = ['otzu', 'isodata', 'minimum']

    #CaCo parameters
    param3 = {  # @@
        'input_dir'           : 'select directory', # input  directory [only photos]
        'output_dir'          : 'select directory', # output directory
        'output_xls'          : 'result_cc.xls',    # name of the output spreadsheet
        'raw_processing'      : True    ,          # set True for raw format input (for example .NEF)

        'band'                : 'grey',       # all CaCo statistics are computed on this band; alternatives are defined in <band choices>
        'band_choices'        : band_choices, # choices available for <band>
        'keys'                : keys,         # optional statistics in output; multiple selection are possible
        'th_choices'          : th_choices,   # choices available for <threshold>
        'threshold'           : 'minimum',    # default thresholding method applied; alternatives are defined in <th_choices>

        'th_switch' : True,      # (True/False) visual output
        'th_dir' : 'th_img',     # directory of the visual output

        'get_tree_id' : False   # add the tree id in the spreadsheet. Use below notation in case
    }

    '''
    notation example to get tree id
    -------------------------------

              abies01_001
              ^       ^
    "tree identifier"_"photo_id"
    '''


    return param3


# @@ CACO WARNINGS
WARN = {
        'ioW': 'CACO is started without selecting input or output directory',
        'bandW': 'selected band is not available',
        }


# @@ PERFORM CHECKS AND INIT PROJECT DIRECTORIES
def init_caco(param3):  #  see get_param3 function
    '''
    Check user input

    returns:
        > list of warning keys if any check fails; see WARN
        > empty list otherwise
    '''
    warning = []

    # ALIAS
    d_msg = 'select directory'
    o_dir = param3['output_dir']
    i_dir = param3['input_dir']

    # CHECK
    if i_dir == d_msg or o_dir == d_msg:
        warning.append("ioW")
    if not param3['band'] in param3['band_choices']:
        warning.append("bandW")

    # DEFINE A DEFAULT OUTPUT XLS SPREADSHEET NAME
    if param3['output_xls'] == 'default':
        out = 'gap_fraction.xls'
        param3['output_xls'] = out
    # add extension if missing
    if not param3['output_xls'].endswith('.xls'):
        param3['output_xls'] = param3['output_xls'] + '.xls'

    # INIT PROJECT DIRECTORIES
    # ========================
    # make output directory
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    # make photo output subdirectory
    if param3['th_switch']:
        try:
            th_dir = os.path.join(o_dir, param3['th_dir'])
            os.mkdir(th_dir)
        except Exception as e:  # directory exists
            pass
    return warning


# @@ ASSESS THE GAP FRACTION FOR EACH IMAGE IN THE INPUT DIRECTORY AND WRITE IT ON XLS
def caco_all(param3):  # param3 defaults are defined in PARAM3
    '''Assess the gap fraction

       input: param3 dictionary
       output: xls spreadsheed with gap fraction
    '''

    warning = init_caco(param3)
    if warning:
        print(warning)

    # ALIAS
    o_dir = param3['output_dir']
    i_dir = param3['input_dir']

    # PARSE ALL THE FILES IN THE INPUT DIRECTORY AND INIT PANDAS TABLE
    data_2   = [(x, 0) for x in os.listdir(i_dir)]
    label_2  = ['filename', 'gap_fraction']
    data = pds.DataFrame.from_records(data_2, columns=label_2)
    for key in param3['keys']:
        data[key] = None

    # EVERYTHING IS READY TO START CACO

    # report elements
    div = '----------------------------------------------------'
    lm = '    '  # left margin
    div = lm + div

    print('\n')
    print(div)
    print(lm + "CaCo started processing:")
    print(div)
    for row in data.index:
        record = data.loc[row]
        # initialize image analysis
        print(lm + ">   {}".format(record.filename))

        photo_path = os.path.join(i_dir, record.filename)
        try:
            # init CaCo analysis for the current image
            caco = CacoImg(param3, photo_path)
            # run CaCo
            caco_data = caco.run()
            # get statistics produced
            if param3['get_tree_id']:
                data.loc[row, 'tree_id'] = record.filename.split('_')[0]
            for key in param3['keys']:
                data.loc[row, key] = caco_data[key]
        except Exception as e:
            print('got oyu')
            print(e)

    # write statistics
    xls_path = os.path.join(param3['output_dir'], param3['output_xls'])
    data.to_excel(xls_path, index=False, sheet_name='cc_results')
    print(div)
    print(lm + 'CaCo analysis is complete!')
    print(lm + 'Results folder: {}'.format(param3['output_dir']))
    print(div)


