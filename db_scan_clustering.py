# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:57:36 2017

@author: Mohammad
"""

import numpy as np
import csv
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_data(filename):
    csvf = open(filename,'rU')
    rows = csv.reader(csvf)
    data = [row for row in rows]
    csvf.close()
    return data

data= read_data('data.csv')
#print(offer_sheet)
#db = DBSCAN(eps= .1, min_samples=100).fit(data)
#print(db.labels_)
kmeans = KMeans(n_clusters=7, random_state=0).fit(data)
#print(kmeans.labels_)



w, h = 7, 7;
clus_score_matrix = [[0 for x in range(w)] for y in range(h)]
                      
#print(clus_score_matrix)                      
                      
def set_cluster_base_score():
    index = 0
    
    for i in kmeans.labels_:
        if i == 0:
            tmpSum = clus_score_matrix[0]+data[index]
            clus_score_matrix[0].append(tmpSum)
            print(tmpSum)
        elif i == 1:
            tmpSum = clus_score_matrix[1]+data[index]
            clus_score_matrix[1].append(tmpSum)
            print(tmpSum)
        elif i == 2:
            tmpSum = clus_score_matrix[2]+data[index]
            clus_score_matrix[2].append(tmpSum)
            print(tmpSum)            
        elif i == 3:
            tmpSum = clus_score_matrix[3]+data[index]
            clus_score_matrix[3].append(tmpSum)
            print(tmpSum)
        elif i == 4:
            tmpSum = clus_score_matrix[4]+data[index]
            clus_score_matrix[4].append(tmpSum)
            print(tmpSum)
        elif i == 5:
            tmpSum = clus_score_matrix[5]+data[index]
            clus_score_matrix[5].append(tmpSum)
            print(tmpSum)
        else:
            tmpSum = clus_score_matrix[6]+data[index]
            clus_score_matrix[6].append(tmpSum)
            print(tmpSum)
        index = index + 1
    return;
                      
set_cluster_base_score()

print(clus_score_matrix)

def plot_cluster(data, n_digits):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    #print(kmeans.labels_)
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    print(reduced_data)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on Yelp data\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return;
    
    
#plot_cluster(data,7)    





