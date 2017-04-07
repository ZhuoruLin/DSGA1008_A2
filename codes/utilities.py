#Import neccesary packages
import torch
import numpy as np
import pandas as pd
import model
import data
import sklearn
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import collections
import gzip
import os

def load_glove_embeddings(glove_directory,emsize=50,voc_size=50000):
    #get directory name glove.6B or other training corpus size
    if glove_directory[-1] =='/':
        dirname = glove_directory.split('/')[-2]
    else:
        dirname = glove_directory.split('/')[-1]
    if emsize in [50,100,300]:
        f = open(os.path.join(glove_directory,'%s.%sd.txt'%(dirname,emsize)))
    else:
        print('Please select from 50, 100 or 300')
        return
    loaded_embeddings = collections.defaultdict()
    for i, line in enumerate(f):
        if i >= voc_size: 
            break
        s = line.split()
        loaded_embeddings[s[0]] = np.asarray(s[1:],dtype='float64')
    return loaded_embeddings

def clusterPlot2D(rep2D,kmeans,savepath = 'TSNE_Plot',title='T-SNE representations'):
	# Set Figure size
    plt.figure(figsize=(20,20))
    #Scatter Plot of 2D representations Generated from t-SNE
    pylab.scatter(rep2D[:,0],rep2D[:,1],s=2,c=kmeans.labels_,cmap=pylab.cm.Accent)
    #Scatter plot of kmeans center
    pylab.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='r')
    # Annotate the kmeans center position
    bbox_props = dict(boxstyle="rarrow", fc="cyan",pad=0.001, ec="b", lw=0.1)
    for center_idx,center in enumerate(kmeans.cluster_centers_):
        pylab.annotate(center_idx,center,bbox=bbox_props,rotation=45,va='top',ha='right')
    plt.title(title)
    plt.savefig('%s.png'%(savepath))
    plt.show()

def zoom(rep2D,kmeans,corpus,index_to_zoom,closest = 100):
	#Extract cluster index for each representation
    cluster_index = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    #Set Figure size
    plt.figure(figsize=(10,10))
    #Extract the subset of certain index
    specified_pos = np.where(cluster_index==index_to_zoom)[0]
    rep2D_to_plot = rep2D[specified_pos]
    #calculate the distances between each word and its cluster center
    center = cluster_centers[index_to_zoom]
    dist2ctr = cdist(center.reshape(-1,2),rep2D_to_plot.reshape(-1,2),'euclidean')[0]
    #Sort by Euclidean Dist
    mat = np.zeros(len(dist2ctr)*4).reshape(len(dist2ctr),4)
    mat[:,:2] = rep2D_to_plot
    mat[:,2] = dist2ctr 
    mat[:,3] = specified_pos
    sorted_mat = np.array(sorted(mat,key=lambda x:x[2]))
    #Extract the N nearest neighbour
    first_N_rep = sorted_mat[:closest,:2]
    first_N_idx = [int(x) for x in sorted_mat[:closest,3]]
    words_to_annotate = [corpus.dictionary.idx2word[idx] for idx in first_N_idx]
    #pylab.scatter(rep2D_to_plot[:,0],rep2D_to_plot[:,1],s=2)
    pylab.scatter(first_N_rep[:,0],first_N_rep[:,1],s=2)
    pylab.scatter(cluster_centers[index_to_zoom][0],cluster_centers[index_to_zoom][1],c='r')
    for word_pos,coordinate in enumerate(first_N_rep):
        pylab.annotate(words_to_annotate[word_pos],coordinate)
    plt.show()
    return words_to_annotate
