#%%
import numpy as np
import random

from PIL import Image

text0 = Image.open("Colored_Brodatz/D75_COLORED.tif")

#%%

#couper l'image en patchs
#faire une fonction qui prend en entrée une image et le nombre de patchs
#et qui renvoie une liste de patchs (nb_patchs * nb_patchs)

def cut_image(image, nb_patchs):
    image = np.array(image)
    patchs = []
    size_patch = image.shape[0]//nb_patchs
    for i in range(nb_patchs):
        for j in range(nb_patchs):
            patchs.append(image[i*size_patch:(i+1)*size_patch, j*size_patch:(j+1)*size_patch])
    return patchs

list_patchs = cut_image(text0, 2)

#On veut garder la moitié des patchs dans l'ensemble D0 et l'autre moitié dans l'ensemble de test DT
#On va donc créer deux listes de patchs

list_patchs_D0 = []
list_patchs_DT = []

for i in range(len(list_patchs)):
    if i%2 == 0:
        list_patchs_D0.append(list_patchs[i])
    else:
        list_patchs_DT.append(list_patchs[i])

#%%

#On va maintenant implémenter l'algorithme de classification k-means
#On va donc créer un dictionnaire de clusters (dictionnaire de listes de patchs)
#On va ensuite créer un dictionnaire de moyennes (dictionnaire de listes de moyennes de patchs)
#On va ensuite créer un dictionnaire de variances (dictionnaire de listes de variances de patchs)
#On va ensuite créer un dictionnaire de distances (dictionnaire de listes de distances de patchs)
#On va ensuite créer un dictionnaire de probabilités (dictionnaire de listes de probabilités de patchs)

#On va créer une fonction qui prend en entrée un patch et un dictionnaire de moyennes
#et qui renvoie la moyenne la plus proche du patch

def closest_mean(patch, dict_means):
    list_means = []
    for key in dict_means.keys():
        list_means.append(dict_means[key])
    list_means = np.array(list_means)
    patch = np.array(patch)
    patch = patch.flatten()
    dist = np.linalg.norm(list_means - patch, axis=1)
    return np.argmin(dist)

#On va créer une fonction qui prend en entrée un patch et un dictionnaire de variances
#et qui renvoie la variance la plus proche du patch

def closest_variance(patch, dict_variances):
    list_variances = []
    for key in dict_variances.keys():
        list_variances.append(dict_variances[key])
    list_variances = np.array(list_variances)
    patch = np.array(patch)
    patch = patch.flatten()
    dist = np.linalg.norm(list_variances - patch, axis=1)
    return np.argmin(dist)

#On va créer une fonction qui prend en entrée un patch et un dictionnaire de distances
#et qui renvoie la distance la plus proche du patch

def closest_distance(patch, dict_distances):
    list_distances = []
    for key in dict_distances.keys():
        list_distances.append(dict_distances[key])
    list_distances = np.array(list_distances)
    patch = np.array(patch)
    patch = patch.flatten()
    dist = np.linalg.norm(list_distances - patch, axis=1)
    return np.argmin(dist)

#On va créer une fonction qui prend en entrée un patch et un dictionnaire de probabilités
#et qui renvoie la probabilité la plus proche du patch

def closest_probability(patch, dict_probabilities):
    list_probabilities = []
    for key in dict_probabilities.keys():
        list_probabilities.append(dict_probabilities[key])
    list_probabilities = np.array(list_probabilities)
    patch = np.array(patch)
    patch = patch.flatten()
    dist = np.linalg.norm(list_probabilities - patch, axis=1)
    return np.argmin(dist)

#On va créer une fonction qui prend en entrée un patch et un dictionnaire de clusters
#et qui renvoie le cluster le plus proche du patch

def closest_cluster(patch, dict_clusters):
    list_clusters = []
    for key in dict_clusters.keys():
        list_clusters.append(dict_clusters[key])
    list_clusters = np.array(list_clusters)
    patch = np.array(patch)
    patch = patch.flatten()
    dist = np.linalg.norm(list_clusters - patch, axis=1)
    return np.argmin(dist)


