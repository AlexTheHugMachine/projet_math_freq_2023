#%%
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os

from PIL import Image

from projet_mfreq_jd_part1 import synthetise_texture

#### FONCTIONS POUR L'EXECUTION KMEANS ####

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

# On doit savoir comment on va représenter nos patchs : par exemple via un histogramme ou filtre, etc..
# on calcule donc l'angle du gradient en chaque pixel du patch et ensuite on le met sous forme d'histogramme pour pouvoir le comparer.
def calcule_histogrammes(patches):
    # Initialiser un tableau pour stocker les angles
    angles = []
    hists = []
    bucket = []

    # Parcourir tous les patches
    for patch in patches:
        imgray = Image.fromarray(patch).convert('L')
        graypatch = np.array(imgray)
        hist = []
        bins = []
        angles = []
        # Calculer les dérivées partielles en x et y
        sobelx = cv.Sobel(graypatch, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(graypatch, cv.CV_64F, 0, 1, ksize=5)

        # Convertir les dérivées en magnitude et angle
        _, angle = cv.cartToPolar(sobelx, sobely, angleInDegrees=True)

        # Ajouter les angles au tableau
        angles.extend(angle.ravel())

        # Construire l'histogramme
        hist, bins = np.histogram(angles, bins=16, range=(0, 360))

        #print('taille du graypatch :', graypatch.shape)
        hists.append(hist)
        bucket.append(bins)
    return hists, bucket


# Il faut ensuite pouvoir comparer des distances entre 2 histogrammes :
#  attention, ils peuvent ne pas avoir le même nombre de pixels
def histograms_distance(hist1, hist2):
    assert len(hist1) == len(hist2), "Les histogrammes doivent avoir la même taille"
    sum1 = np.sum(hist1) # Si patchs de même taille sum1 = sum2
    sum2 = np.sum(hist2)
    
    # On normalise avec la somme des pixels du patch
    dist = np.sum(((hist1/sum1) - (hist2/sum2))**2)
    return dist

# Le centroids sont des listes d'histogrammes : on doit pouvoir les comparer
def compare_centroids(liste1, liste2):
    if len(liste1) != len(liste2):
        return False

    for sous_liste1, sous_liste2 in zip(liste1, liste2):
        if sous_liste1 != sous_liste2:
            return False

    return True

# Fonction pour tirer aléatoirement des centroids dans les données
# On lui passe la liste des histogrammes en donnees et les labels associés aux patchs
def random_centroids(donnees, labels, N):
    centroids = []
    centroids_labels = []
    indices_disponibles = [i for i in range(len(donnees))]
    for i in range(N):
        id = random.choice(indices_disponibles)
        centroids.append(donnees[id].tolist())
        centroids_labels.append(labels[id])
        indices_disponibles.remove(id)
    return centroids, centroids_labels

# Fonction pour classifier un ensemble de données dans les clusters représentés par les centroïdes
#     donnees est la liste contenant les objets à classifier
#     hists est l'histogramme des angles pour chaque objet
#     bucket représente un tableau contenant des informations sur les angles stockés dans hists
#     labels est la liste des id des textures d'où provient chaque objet
#     N est le nombre de clusters de l'algo
# centroid contient les N valeurs moyennes des histogrammes dans chaque cluster
def classifier(donnees, hists, bucket, labels, N, centroids):
    # Pour cette fonction, on classe juste les données en fonction de leur proximités avec les clusters :
    # On ne met pas à jour les clusters

    # Créer un tableau de listes pour stocker les patches de chaque cluster
    clusters = [[] for _ in range(N)]
    hist_clusters = [[] for _ in range(N)]
    labels_clusters = [[] for _ in range(N)]

    #On parcours tous les patchs
    for d in range(len(donnees)):
        #print("On parcourt le patch n° ", d)
        # Pour l'histogramme d'indice d, on cherche quel centroid est le plus proche :
        min_distance = 100000000.0
        indice_min_distance = -1
        # Boucle pour comparer l'histogramme donné avec chaque centroïde
        for i, centroid in enumerate(centroids):
            distance = histograms_distance(hists[d], centroid)
            if distance < min_distance:
                min_distance = distance
                indice_min_distance = i

        if indice_min_distance != -1:
            #print("donnee n°", d, "  ajoutée au cluster n°", indice_min_distance)
            clusters[indice_min_distance].append(donnees[d])
            hist_clusters[indice_min_distance].append(hists[d])
            labels_clusters[indice_min_distance].append(labels[d])
        else :
            print("Aucun centroid ne minimise la distance (?)")

    return clusters, hist_clusters, labels_clusters


# La fonction pour faire une itération de KMeans
def kmeans(donnees, hists, bucket, labels, N, centroids):
    # On clusterise les données en fonction des centroids donnés
    clusters, hist_clusters, labels_clusters = classifier(donnees, hists, bucket, labels, N, centroids)

    # On recalcule les centroides de chaque cluster :
    for c in range(N):
        if len(clusters[c]) != 0:
            tmp_hist = np.mean(hist_clusters[c], axis=0)
            centroids[c] = tmp_hist.tolist()

    # Affichage final de la taille des clusters :
    '''for c in range(N):
        print("Taille cluster ", c, " : ", len(clusters[c]))
        #print(hist_clusters[c])'''

    return clusters, labels_clusters, centroids

######################################################################### ETAPE 1 : INITIALISATION
# charger les images dans le répertoire :
chemin = "Colored_Brodatz/"
prefix = "D"
suffix = "_COLORED.tif"

# Définir la plage de numéros des images que vous voulez charger
start_number = 3 
end_number = 8 #112

N = (end_number+ 1) - start_number

patches = []
centroids_init = []
labels_init = []
buck = []

# Rassembler la moitié des patchs dans train et l'autre dans test
patchs_DO = []
patchs_DT = []
labels_DO = []
labels_DT = []

# nombre de division en absisse et ordonnées:
div = 6

print("Nombre de patchs par texture : ", div*div)

# Charger les images
for i in range(start_number, end_number + 1):
    file_path = f"{chemin}{prefix}{i}{suffix}"
    img = Image.open(file_path)

    # On découpe l'image en patchs qu'on stocke dans une liste
    liste_patchs = cut_image(img, div)
    patches.extend(liste_patchs)

    # une partie va dans DO et l'autre dans DT :
    list_DO = []
    list_DT = []
    for p in range(len(liste_patchs)):
        if p%2 == 0:
            list_DO.append(liste_patchs[p])
            labels_DO.append(i)
        else :
            list_DT.append(liste_patchs[p])
            labels_DT.append(i)
    
    # On propose un centroide aléatoire par texture:
    tmphistsDO, buck = calcule_histogrammes(list_DO)
    centroids_init.append(random.choice(tmphistsDO).tolist())
    labels_init.append(i)

    patchs_DO.extend(list_DO)
    patchs_DT.extend(list_DT)
    

print("Taille jeux apprentissage DO : ", len(patchs_DO))
print("Taille jeux test DT : ", len(patchs_DT))
print("Nombre de Clusters N : ", N)

hist_DO, bucket_DO = calcule_histogrammes(patchs_DO)

# INITIALISATION DES CENTROIDES :
# Aléatoirement parmis tous les patchs DO : risque d'avoir plusieurs centroïdes centrés sur une même texture
# Risque de convergence vers un minimum local

''' On peut soit prendre les centroids aleatoirement dans tous les patch : '''
centroids_DO, centroids_labels_DO = random_centroids(hist_DO, labels_DO, N)

'''Ou aléatoirement dans chaque texture differente (fait à l'initialisation)
    # On pourrait pour avoir des centroides plus fidèles choisir aléatoirement un fragment
    # des données lorsqu'elles sont découpées, afin que chaque centroide soit plus représentatif
    # du cluster et éviter que certaines données représentent plusieurs clusters simultannément'''
#centroids_DO = centroids_init
#centroids_labels_DO = labels_init


# On fait U fois la classification KMeans pour terminer l'algorithme s'il ne converge pas
U = 10
iteration = 0
convergence = 0

clusters = [[] for _ in range(N)]
labels_clusters = [[] for _ in range(N)]

while iteration < U:
    print("iteration ", iteration)
    last_centroids = centroids_DO.copy()
    clusters, labels_clusters, centroids_DO = kmeans(patchs_DO, hist_DO, bucket_DO, labels_DO, N, centroids_DO)
    
    if last_centroids == centroids_DO:
        convergence += 1
        if convergence >= 3:
            print("L'algorithme a convergé ! On s'arrête là")
            break
    else :
        convergence = 0

    iteration += 1

# On veut maintenant afficher quelles textures on été retenues pour chaque Cluster : 
'''for n in range(N):
    # Créer une figure et organiser les sous-graphiques
    plt.figure(figsize=(10,6))

    for e in range(len(clusters[n])):
        nb_val = len(clusters[n])
        div = math.ceil(math.sqrt(nb_val))
        # Ajouter le premier sous-graphique
        plt.subplot(div, div, e+1)
        plt.imshow(clusters[n][e], cmap='gray')
        plt.title(f'Fragment de texture n° {labels_clusters[n][e]}')

    # Afficher la figure
    plt.tight_layout()
    plt.show()'''

# On cree une fonction qui retourne les elements uniques d'une liste : pour savoir combien de valeurs de label differentes elle contient
def unique(list1):
  
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

# Cette fonction doit être appelée après l'apprentissage du KMeans sur DO, elle renvoie les
#  labels de chaque centroïde en fonction du plus grand nombre de texture d'une même texture en son sein.
#  Cette même fonction ne doit PAS être appelée sur un autre jeu que DO, surtout pas avant de classifier
#  un autre jeu de donnée.
def trouve_texture_centoides(clusters, labels_clusters, centroid_labels, N):
    for n in range(N):    
        # On cherche la plus grande occurence d'un label de texture pour l'attribuer au cluster n
        label_list = unique(labels_clusters[n])

        if len(label_list) != 0:
            id_cluster = 0
            for f in range(len(label_list)) : 
                if labels_clusters[n].count(label_list[f]) > labels_clusters[n].count(label_list[id_cluster]):
                    id_cluster = f

            centroid_labels[n] = label_list[id_cluster]

    
# On veut à présent donner le pourcentage de textures qui ont mal été classifiées.
#       clusters : une liste de taille N, contenant d'autre listes d'objets pour chaque clusters.
#       labels_clusters : une liste de taille N, qui contient pour chaque element de chaque cluster,
#                   l'id du label d'où il provient.
#       centroid_labels : une liste de taille N, qui stocke de quel id de texture provient la 
#                   moyenne de chaque cluster au départ.
#       N : nb de clusters de l'algo.
def calcule_accuracy(clusters, labels_clusters, centroid_labels, N):
    percent_clusters = np.zeros(N)
    for n in range(len(clusters)):
        # Pour chaque cluster on fait un pourcentage de reussite:
        nb_elem = len(clusters[n])
        nb_bons = 0

        #print("CLUSTER N°", n, ", dont le centroid est basé sur texture n°", centroid_labels[n], " : ")
        for e in range(len(clusters[n])):
            # Pas une bonne méthode pour déterminer quelle texture est représentée dans le cluster
            #print("     ", labels_clusters[n][e])
            if labels_clusters[n][e] == centroid_labels[n] :
                nb_bons += 1
        #print("Nb elements dans cluster ", n , " : ", len(clusters[n]))
        #print("nb bons : ", nb_bons)
        if len(labels_clusters[n]) != 0 :
            pourcent = float(nb_bons / len(labels_clusters[n]))
            percent_clusters[n] = pourcent
        else :
            print("Division par zero : cluster vide")
            percent_clusters[n] = 1.0
        #print("Label pour ce cluster : ", centroid_labels[n])
        #print("Pourcentage de bonnes textures retrouvées pour le cluster ", n, " : ", pourcent)
        
    moyenne_accuracy = sum(percent_clusters) / len(clusters)
    print("Tableau accuracy : ", percent_clusters)
    print("Accuracy totale : ", moyenne_accuracy)

# On labelise les clusters en choisissant le label de texture le plus présent pour chaque cluster    
trouve_texture_centoides(clusters, labels_clusters, centroids_labels_DO, N)
print(" ")
print(" ")
print(" ****************** ********* *******************")
print("Classification des données DO :")
calcule_accuracy(clusters, labels_clusters, centroids_labels_DO, N)
print(" ****************** ********* *******************")
print(" ")


# Maintenant, on veut utiliser les centroids mis-à-jour pour classifier la partie des patchs Tests:
hists_DT, bucket_DT = calcule_histogrammes(patchs_DT)
clusters_DT, _, labels_clusters_DT = classifier(patchs_DT, hists_DT, bucket_DT, labels_DT, N, centroids_DO)

print(" ")
print(" ")
print(" ****************** ********* *******************")
print("Classification des données DT :")
calcule_accuracy(clusters_DT, labels_clusters_DT, centroids_labels_DO, N)
print(" ****************** ********* *******************")
print(" ")

# On veut créer de nouvelles textures à partir des patchs de DT avec notre générateur de textures
# pour ensuite les classifier et vérifier qu'elles sont attribuées à la bonne texture :


################################ ON SYNTHETISE LES TEXTURES
# On crée une liste de patchs choisis aléatoirement dans DT de taille T:
T = 8
patch_reconstit = []
reconstit_labels = []
indices_disponibles = [i for i in range(len(patchs_DT))]
for i in range(T):
    id = random.choice(indices_disponibles)
    patch_reconstit.append(patchs_DT[id])
    reconstit_labels.append(labels_DT[id])
    indices_disponibles.remove(id)
 

# On synthetise des textures à partir des patchs stockés dans patch_reconstit
textures = []


# Le code donné est pour démontrer la procédure de création de textures générées
#  à partir des patchs des DT, mais à cause du temps d'éxécution de notre algorithme 
#  de génération : nous avons donc sauvegardé des résultats dans un dossier DT_textures
#  pour les textures n° 1 à 12, que nous récupérons plus bas.
'''for t in range(len(patch_reconstit)):
    # faire appel à la fonction de synthetisation de texture.
    print("synthetise texture n°", t)
    patch = Image.fromarray(patch_reconstit[t]).convert('RGB')
    print("shape de patch : ", patch.size)
    #assert 
    #texture = synthetise_texture(patch, 32, 10, 16)
    taille_patch = math.ceil(max(patch.size[0], patch.size[1]) / 3)
    if taille_patch%2 != 0:
        taille_patch -= 1
    #texture = synthetise_texture(patch, max(patch.size[0], patch.size[1]), 12, 22)
    texture = synthetise_texture(patch, 64, 10, 32)
    prefixe_dt = "DT_textures/texture_generee_DT_"
    num = ""
    it = 0
    suffixe_dt = ".png"
    while os.path.exists(f"{prefixe_dt}{reconstit_labels[t]}{num}{suffixe_dt}"):
        num = "_" + f"{it}"
        it += 1
    texture.save(f"{prefixe_dt}{reconstit_labels[t]}{num}{suffixe_dt}")
    texture.show()
    textures.append(texture)'''


########################### ON CHARGE LES TEXTURES
# Algo qui charge des textures synthétisées avec un n° de label correspondant
synth_patchs = []
synth_patchs_labels = []    

list_id_text = unique(centroids_labels_DO)
for num_label in list_id_text:
    synth_path = "DT_textures/texture_generee_DT_"
    suffix_synth = ".png"
    if os.path.exists(f"{synth_path}{num_label}{suffix_synth}"):
        txt = Image.open(f"{synth_path}{num_label}{suffix_synth}").convert('RGB')
        liste_patchs_synth = cut_image(txt, 3)
        synth_patchs.extend(liste_patchs_synth)

        for h in range(len(liste_patchs_synth)):
            synth_patchs_labels.append(num_label)

    else :
        print("La texture généréee n°", num_label, " n'existe pas dans DT_textures.")

# On va maintenant vérifier que les patchs dans synth_patchs sont bien classifiés par notre KMeans entrainé
# Maintenant, on veut utiliser les centroids mis-à-jour pour classifier la partie des patchs Tests:
hists_synthese, bucket_synthese = calcule_histogrammes(synth_patchs)
clusters_synthese, _, labels_clusters_synthese = classifier(synth_patchs, hists_synthese, bucket_synthese, synth_patchs_labels, N, centroids_DO)

# On affiche quelles textures on été retenues pour chaque Cluster : 
'''for n in range(N):
    # Créer une figure et organiser les sous-graphiques
    plt.figure(figsize=(10,6), num=f"Cluster n°{n} avec le label {centroids_labels_DO[n]}")
    print("Cluster n°", n, " a ", len(clusters_synthese[n]), " élements !")
    
    for el in range(len(clusters_synthese[n])):
        nb_val = len(clusters_synthese[n])
        div = math.ceil(math.sqrt(nb_val))
        # Ajouter le premier sous-graphique
        plt.subplot(div, div, el+1)
        plt.imshow(clusters_synthese[n][el], cmap='gray')
        plt.title(f'Fragment de texture n° {labels_clusters_synthese[n][el]}')

    # Afficher la figure
    plt.tight_layout()
    plt.show()'''

print(" ")
print(" ")
print(" ****************** ********* *******************")
print("Classification des textures synthétisées à partir de DT :")
calcule_accuracy(clusters_synthese, labels_clusters_synthese, centroids_labels_DO, N)
print(" ****************** ********* *******************")


# On charge des images bruitées et floues pour voir si elles sont bien classifiées 
# avec les centroïdes de DO :

patches_blur = []
patches_big_blur = []
patches_noise = []

modified_patches = []
modified_patches_labels = []

# On charge toutes les images bruitees et on oublie pas de leur donner un label associé
#  à la texture qu'on a floutée
for i in range(15) : # range(15) car on a principalement travaillé sur les 15 premieres textures du folder
    noise_path = "flou/texture_generee_DT_"
    suffix_blur = "_blur"
    suffix_big_blur = "_blur_2"
    suffix_noised = "_bruit"
    extension = ".tif"

    # Existe-t-il une version floue de la texture i ?
    if os.path.exists(f"{noise_path}{i}{suffix_blur}{extension}"):
        im_blur = Image.open(f"{noise_path}{i}{suffix_blur}{extension}").convert('RGB')
        patches_blur = cut_image(im_blur, 3)
        modified_patches.extend(patches_blur)
        for j in range(len(patches_blur)):
            modified_patches_labels.append(i)
    # Existe-t-il une version très floue de la texture i ?
    if os.path.exists(f"{noise_path}{i}{suffix_big_blur}{extension}"):
        im_big_blur = Image.open(f"{noise_path}{i}{suffix_big_blur}{extension}").convert('RGB')
        patches_big_blur = cut_image(im_big_blur, 3)
        modified_patches.extend(patches_big_blur)
        for j in range(len(patches_big_blur)):
            modified_patches_labels.append(i)
    # Existe-t-il une version bruitée de la texture i ?
    if os.path.exists(f"{noise_path}{i}{suffix_noised}{extension}"):
        im_noise = Image.open(f"{noise_path}{i}{suffix_noised}{extension}").convert('RGB')
        patches_noise = cut_image(im_noise, 3)
        modified_patches.extend(patches_noise)
        for j in range(len(patches_noise)):
            modified_patches_labels.append(i)

# On va maintenant vérifier que les patchs dans textures sont bien classifier par notre KMeans entrainé
# Maintenant, on veut utiliser les centroids mis-à-jour pour classifier la partie des patchs Tests:
hists_modified, bucket_modified = calcule_histogrammes(modified_patches)
clusters_modified, _, labels_clusters_modified = classifier(modified_patches, hists_modified, bucket_modified, modified_patches_labels, N, centroids_DO)

print(" ")
print(" ")
print(" ****************** ********* *******************")
print("Classification des textures floutées et bruitées :")
calcule_accuracy(clusters_modified, labels_clusters_modified, centroids_labels_DO, N)
print(" ****************** ********* *******************")

for n in range(N):
    # Créer une figure et organiser les sous-graphiques
    plt.figure(figsize=(10,6), num=f"Cluster n°{n} avec le label {centroids_labels_DO[n]}")
    #print("Cluster n°", n, " a ", len(clusters_modified[n]), " élements !")
    
    for el in range(len(clusters_modified[n])):
        nb_val = len(clusters_modified[n])
        div = math.ceil(math.sqrt(nb_val))
        # Ajouter le premier sous-graphique
        plt.subplot(div, div, el+1)
        plt.imshow(clusters_modified[n][el], cmap='gray')
        plt.title(f'Fragment de texture n° {labels_clusters_modified[n][el]}')

    # Afficher la figure
    plt.tight_layout()
    plt.show()