# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:03:20 2018

@author: jmduarte
"""
import os
from PIL import Image
import spectral
import numpy
from sklearn.ensemble import RandomForestClassifier
import scipy.ndimage
import math
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load

#########################
# MODEL SELECTION
#########################
root = 'D:/Escritorio/Corpoica/Canopy Attributes/images/'
root_in = root + 'input/'
root_out = root + 'output/'
L = os.listdir(root_in)     
cont = 0
for l in range(len(L)):
    file = L[l]
    file_split = os.path.splitext(file)
    if file_split[1] == '.gis':
        labelsFile = root_in + file
        local_labels = spectral.open_image(labelsFile).read_band(0)
        no_classes = int( local_labels.max() )
        for i in range(1, no_classes+1):
            (rows, cols)=(local_labels == i).nonzero()
            samples = len(rows)
            cont += samples
 
data = numpy.zeros((cont,3), dtype=numpy.float32)           
labels = numpy.zeros((cont,1), dtype=numpy.uint8)            
labels = labels.flatten()

cont = 0            
for l in range(len(L)):
    file = L[l]
    file_split = os.path.splitext(file)
    if file_split[1] == '.gis':
        labelsFile = root_in + file
        local_labels = spectral.open_image(labelsFile).read_band(0)
        image = file_split[0] + '.jpg'
        imageFile = root_in + image
        img = numpy.asarray( Image.open(imageFile), dtype=numpy.float32 )
        no_classes = int( local_labels.max() )
        for i in range(1, no_classes+1):
            (rows, cols)=(local_labels == i).nonzero()
            samples = len(rows)
            for p in range(samples):
                data[cont,:] = img[rows[p],cols[p], :].flatten()
                labels[cont] = i-1
                cont += 1

data /= 255

crossval_splits = 5
accuracy = numpy.zeros(crossval_splits)
sensitivity = numpy.zeros(crossval_splits)
specificity = numpy.zeros(crossval_splits)
cont = 0

skf = StratifiedKFold(n_splits=crossval_splits, shuffle=True, random_state=123)
skf.get_n_splits(data, labels)

for train_index, test_index in skf.split(data, labels):
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    #Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=0, verbose=5)
    model.fit(train_data, train_labels)  
    
    #Compute scores
    pred = model.predict(test_data)
    ConfussionMatrix = numpy.zeros((no_classes,no_classes))
    for i in range(pred.shape[0]):
        ConfussionMatrix[test_labels[i], pred[i]] += 1.0
    
    for i in range(no_classes):
        accuracy[cont] += ConfussionMatrix[i,i]    
    accuracy[cont] /= ConfussionMatrix.sum()
    print('accuracy: ' + str(accuracy[cont]))
    
    for i in range(no_classes):
        den = ConfussionMatrix[:,i].sum()
        if den>0: sensitivity[cont] += ConfussionMatrix[i,i] / den
    sensitivity[cont] /= no_classes
    print('sensitivity: ' + str(sensitivity[cont]))    
    
    for i in range(no_classes):
        den = ConfussionMatrix[i,:].sum()
        if den>0: specificity[cont] += ConfussionMatrix[i,i] / den
    specificity[cont] /= no_classes
    print('specificity: ' + str(specificity[cont])) 
    cont += 1

print('accuracy_mean: ' + str(accuracy.mean()*100))
print('accuracy_std: ' + str(accuracy.std()*100/math.sqrt(crossval_splits)))
print('sensitivity_mean: ' + str(sensitivity.mean()*100))
print('sensitivity_std: ' + str(sensitivity.std()*100/math.sqrt(crossval_splits)))
print('specificity_mean: ' + str(specificity.mean()*100))
print('specificity_std: ' + str(specificity.std()*100/math.sqrt(crossval_splits)))

#########################
# CLASSIFICATION
#########################
model = RandomForestClassifier(n_estimators=100, random_state=0, verbose=5)
model.fit(data, labels)  
dump(model, 'D:/Escritorio/Corpoica/Canopy Attributes/RandomForest.joblib')
    
#######################
# CANOPY ATTRIBUTES
#######################
model = load('D:/Escritorio/Corpoica/Canopy Attributes/RandomForest.joblib')
output = open(root_out + 'Canopy Attributes_Random Forest.csv', 'w')
output.write('image; gap_fraction; large_gap_fraction; foliage cover; '
             'crown cover; crown porosity; clumping index; LAI\n')
for l in range(len(L)):
    file = L[l]
    file_split = os.path.splitext(file)
    if file_split[1] == '.jpg':
        #Performs classification on all the images
        imageFile = root_in + file
        img = numpy.asarray(Image.open(imageFile), dtype=numpy.float32)
        rows = img.shape[0]
        cols = img.shape[1]
        img_flat = numpy.asarray(img.reshape(rows*cols,3), dtype=numpy.float32)
        img_flat /= 255
        classify = model.predict(img_flat)
        classify = numpy.asarray( classify.reshape((rows,cols)), dtype=numpy.uint8)
        image_array = numpy.zeros((rows,cols,3), dtype=numpy.uint8)
        (u, v) = (classify == 0).nonzero()  # Trunk 
        if len(u)>0: #Yellow
            image_array[u,v,1] = 127    #Green
            image_array[u,v,0] = 127    #Red
        (u, v) = (classify == 1).nonzero()  # Sky 
        if len(u)>0: image_array[u,v,2] = 127   #Blue
        (u, v) = (classify == 2).nonzero()  # Leaves 
        if len(u)>0: image_array[u,v,1] = 127   #Blue
        image = Image.fromarray(image_array)
        image.save(root_out + file_split[0] + '_RandomForest_class.tif')
        
        #Compute Canopy Attributes
        sky = numpy.asarray( numpy.zeros((rows,cols)), dtype=numpy.uint8 )
        (x,y) = (classify == 1).nonzero()
        sky[x,y] = 1
        region, id = scipy.ndimage.measurements.label(sky)
        # trick 1: img to vector
        region = region.flatten()
        shape_flatten = region.shape
        # trick 2: get the direct mapping of pixel having the the same label [see previous step]
        sort_index = numpy.argsort(region)
        #--------------------------------------------------------------------------------
        # APPLY DIRECT MAPPING TO SAMPLE THE SIZE OF EACH IMG REGION
        sorted_region = region[sort_index]
        img = sky.flatten()[sort_index]
        #----------------------------------------------------------------------------
        # TWO LOOPS ARE PERFOMED:
        # 1. to define the threshold dividing normal gaps from big gaps;
        # 2. to sample the size of the gaps for the normal gaps
        gap_size = []  # big and normal gaps
        start = 0
        img_end = shape_flatten[0] - 1 # total of pixels -1
        np_sum = numpy.sum  # small trick for speed
        # -->> First loop - all gaps
        while start < img_end:
          fid = sorted_region[start]  # fid is equal to the new label value [remember get_regions?]
          end = start
          if end == img_end: break
          # --------------------------------------------------------------------------------
          # end continues to move 1 step ahead while the label is equal
          while end < img_end and fid == sorted_region[end]:
            end += 1 # end finishes one step beyond the edge
          # --------------------------------------------------------------------------------
          if fid > 0:  # this is not vegetation
             stat = np_sum(img[start:end])  # start and end are at the extremes of one label [end is 1 step beyond]
             gap_size.append(stat)
          start = end # end was already on a new label value
        
        #Compute statistics                
        gap_size = numpy.array(gap_size)
        gap_mean = gap_size.mean()
        gap_std_dev = gap_size.std()
        gap_std_err = float(gap_std_dev) / float(len(gap_size))**0.5
        big_gap_size_th = gap_mean + gap_std_err
        gap_px = gap_size.sum() #sum of all gaps
        
        # -->> Second loop - big gaps
        big_gap_size = []
        normal_gap_size = []
        start = 0
        while start < img_end:
          fid = sorted_region[start]  # fid is equal to the new label value [remember get_regions?]
          end = start
          if end == img_end: break
          # --------------------------------------------------------------------------------
          # end continues to move 1 step ahead while the label is equal
          while end < img_end and fid == sorted_region[end]:
            end += 1 # end finishes one step beyond the edge
          # --------------------------------------------------------------------------------
          if fid > 0:  # this is not vegetation
            stat = np_sum(img[start:end])  # start and end are at the extremes of one label [end is 1 step beyond]  
            if stat > big_gap_size_th: big_gap_size.append(stat)
            else: normal_gap_size.append(stat)
          start = end # end was already on a new label value
        
        # normal gap 
        normal_gap_size = numpy.array(normal_gap_size)
        normal_gap_px = normal_gap_size.sum()
        
        # big gap 
        big_gap_size = numpy.array(big_gap_size)
        big_gap_px = big_gap_size.sum()
        
        #image pixels
        image_px = sky.shape[0] * sky.shape[1] 
        #Trunk pixels
        trunk_px = (classify == 0).sum()
        #leaves pixels
        leaf_px =  (classify == 2).sum()  
        
        # compute canopy attributes
        extinction_coeff = 0.5
        gap_fraction = float(gap_px) / float(image_px-trunk_px)
        large_gap_fraction = float(big_gap_px) / float(image_px-trunk_px)
        foliage_cover = float(leaf_px) / float(image_px-trunk_px)
        crown_cover = 1 - large_gap_fraction
        crown_porosity = 1 - (foliage_cover / crown_cover)
        clumping_index = (1 - crown_porosity) * math.log(1 - foliage_cover) \
                       / (foliage_cover * math.log(crown_porosity))
        LAI = -crown_cover * math.log(crown_porosity) / extinction_coeff
        #Write canopy attributes to file
        output.write(file_split[0] + '; ' + str(gap_fraction) + '; ' + \
                     str(large_gap_fraction) + '; ' + str(foliage_cover) + '; ' + \
                     str(crown_cover) + '; ' + str(crown_porosity) + '; ' + \
                     str(clumping_index) + '; ' + str(LAI) + '\n')
output.close()
