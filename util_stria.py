import numpy as np
import os
import cv2
from skimage.io import imsave, imread
import json
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import data
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from scipy import stats as st
from skimage import util
from skimage import morphology
import feret

def areasandecc(testlabel,testlabel2):
    striaarea=np.sum(testlabel)
    totalcaparea=np.sum(testlabel2)
    avgareapercap=0 
    averageecc=0
    if(totalcaparea>0):     
        regions=regionprops(label(testlabel2))
        areas=[]
        ecc=[]
        for i in range(0,len(regions)):
            areas.append(regions[i].area)
            ecc.append(regions[i].eccentricity)
        avgareapercap=np.mean(areas)
        averageecc=np.mean(ecc)
    return striaarea,totalcaparea,avgareapercap,averageecc

def kmeanstria(test,testlabel):
    test=test * testlabel[..., None]
    pixel_values = test.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.2)
    gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    k = 4
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    centers = np.uint8(centers)
    labelim=labels.reshape([512,512])
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(test.shape)
    # imsave('114mask.png',labelim)
    # show the image
    # plt.imshow(labelim)
    # plt.show()
    return labelim,gray,segmented_image

def calcmaxminlabel(gray,labelim,k): 
    region=gray
    val=np.zeros((k,1))
    for i in range (0,k):
        val[i]=np.float32(np.mean(region[labelim==i]))
    
    valid_idx = np.where(val > 1)[0]
    minv = valid_idx[val[valid_idx].argmin()]
    maxv = valid_idx[val[valid_idx].argmax()]
    return minv,maxv

def countcells(cell):
    lab_image = label(cell)
    table = regionprops_table( lab_image, properties=('label', 'area'))
    condition = (table['area'] > 10) 
    input_labels = table['label']
    output_labels = input_labels * condition
    # print(input_labels.shape)
    filtered_lab_image = util.map_array(lab_image, input_labels, output_labels)
    filtered_lab_image = filtered_lab_image>0
    # plt.imshow(filtered_lab_image)
    # plt.savefig('test.png')
    # plt.show()
    lab_image2 = label(filtered_lab_image)
    regions=regionprops(lab_image2)
    areas=[]
    for i in range(0,len(regions)):
        if(regions[i].area<50): 
            areas.append(regions[i].area)
    # print(np.mean(areas))
    numcells=np.sum(filtered_lab_image)/np.mean(areas)
    return round(numcells)

def width(labelimag):
    lab_image = label(labelimag)
    regions=regionprops(lab_image)
    minr, minc, maxr, maxc = regions[0].bbox
    w1=np.sum(labelimag)/np.max([maxr-minr,maxc-minc])
    w2=feret.min(labelimag)
    long=np.argmax([maxr-minr,maxc-minc])
    if(long==0):
        mid=round((maxr-minr)/2)
        w3=np.sum(labelimag[mid][:])
    else:
        mid=round((maxc-minc)/2)
        w3=np.sum(labelimag[:][mid])
    return w1,w2,w3

def capcalc(gray,cap,area,intensity,ecc,solid):
    dia = morphology.diamond(radius=2)
    dia1 = morphology.diamond(radius=1)
    cap=morphology.binary_opening(cap,dia)
    # cap=morphology.binary_closing(cap,dia1)
    lab_image = label(cap)
    table = regionprops_table( lab_image,intensity_image=gray, properties=('label', 'area','solidity','eccentricity','intensity_mean'))
    condition = (table['area'] > area) & (table['intensity_mean'] > intensity) & (table['eccentricity'] < ecc)& (table['solidity'] > solid)
    input_labels = table['label']
    output_labels = input_labels * condition
    # print(input_labels.shape)
    filtered_lab_image = util.map_array(lab_image, input_labels, output_labels)
    filtered_lab_image = filtered_lab_image>0
    return filtered_lab_image
    # plt.imshow(test)
    # plt.imshow(filtered_lab_image,cmap='jet',alpha=0.5)
    # plt.show()
    # lab_image2 = label(filtered_lab_image)
    # regions=regionprops(lab_image2)
    # areas=[]
    # for i in range(0,len(regions)):
    #     if(regions[i].area<50): 
    #         areas.append(regions[i].area)
    # # print(np.mean(areas))
    # numneurons=np.sum(filtered_lab_image)/np.mean(areas)
    # return round(numneurons)
def returnlargarea(mask,loc):
    lab_image = label(mask)
    table = regionprops_table(lab_image, properties=('label', 'area'))
    if(table['area'].size==1):
        return mask
    else:
        condition = (table['area'] > np.min(table['area']))
        input_labels = table['label']
        output_labels = input_labels * condition
        # print(input_labels.shape)
        filtered_lab_image = util.map_array(lab_image, input_labels, output_labels)
        filtered_lab_image = filtered_lab_image>0
        imsave(loc,np.uint8(filtered_lab_image))
        return filtered_lab_image
    
