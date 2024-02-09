# Stria Analysis Overview 
This repository has three jupyter notebooks for to analyze the stria vascularis using the methods proposed in this paper(link). Please download these files as a zip and follow the instruction within the jupyter notebooks! Sample data for analysis can be found here[link](https://drive.google.com/drive/folders/13iQc_jnJShfzLUV10otDVjkDZofoUn6X?usp=share_link) 
Trained nnUnet model can be found here[link](https://drive.google.com/drive/folders/1tRSyyQXr8idvOJnDc-SphXVKj-HfFRdd?usp=share_link)
Please install this after instally pytorch, nnUNet, and Ultralytics before running these notebooks. 
This notebook is designed to work on a CUDA GPU

## preprocess.ipynb
RUN THIS FIRST
This notebook takes the raw whole slide image(WSI) and squares the image using importcv2.py; this notebook also creates the file repository needed for the rest of the project

## generatepatches.ipynb
RUN THIS SECOND
Make sure ultralytics is already installed prior to running the notebook
This notebook takes the preprocessed data and creates 256x256 patches around the stria vascularis using 2 YOLO networks in series 

## runinference.ipynb
RUN THIS THIRD
Make sure nnUnet and pytorch are already installed and working. This notebook calculates the binary masks for the outline of the stria and its associated vasculature

## stria_analysis.ipynb
RUN THIS FOURTH
This notebook analyzes the binary masks generated in the previous step



