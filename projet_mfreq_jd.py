#%%
import numpy as np
import random

from PIL import Image

text0 = Image.open("textures_data/text0.png")
text1 = Image.open("textures_data/text1.png")
text2 = Image.open("textures_data/text2.png")
text3 = Image.open("textures_data/text3.png")
text4 = Image.open("textures_data/text4.png")
text5 = Image.open("textures_data/text5.png")
text6 = Image.open("textures_data/text6.png")

#%%

#text0.show()
#text1bis = np.array(text1)
#print(text1bis.shape)

def ssd_pixel(pixel1, pixel2):
    difference = pixel1 - pixel2
    squared_difference = difference ** 2
    return squared_difference

# faire une fonction qui prenne en entree une image couleur ou grayscale
# donc tester le nombre de channel et traiter l'image en fonction
# Paramètres : (Ismp, size_final, taille_patch (doit etre impair), epsilon)
# Définir une fonction pour calculer la similarité entre deux patches
def patch_similarity(patch1, patch2, epsilon):
    diffR = 0
    diffG = 0
    diffB = 0
    nbpix = 0
    for i in range(patch1.shape[0]):
        for j in range(patch1.shape[1]):
            if patch1[i,j,0] != -1 and patch2[i,j,0] != -1:
                '''diffR += ssd_pixel(patch1[i,j,0], patch2[i,j,0])
                diffG += ssd_pixel(patch1[i,j,1], patch2[i,j,1])
                diffB += ssd_pixel(patch1[i,j,2], patch2[i,j,2])'''
                #print(patch1[i,j,0] - patch2[i,j,0])
                diffR += abs(patch1[i,j,0] - patch2[i,j,0])
                diffG += abs(patch1[i,j,1] - patch2[i,j,1])
                diffB += abs(patch1[i,j,2] - patch2[i,j,2])
                nbpix += 1
    if diffR/nbpix < epsilon and diffG/nbpix < epsilon and diffB/nbpix < epsilon:
        return True
    else: return False

def random_patch(Ismp, size_patch):
    Ismp = np.array(Ismp)
    patch = np.zeros((size_patch, size_patch))
    random_x = np.random.randint(0, Ismp.shape[0] - size_patch)
    random_y = np.random.randint(0, Ismp.shape[1] - size_patch)
    patch = Ismp[random_x:random_x + size_patch, random_y:random_y + size_patch]
    return patch

def extract_patch(image, size, i_centre, j_centre):
    patch = np.full((size, size, 3), -1, dtype=np.int32)
    offset = int((size - 1)/2)
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            patch[offset + j, offset + i] = image[int(j_centre + j), int(i_centre + i)]
    return patch

def image_initiale(Ismp, size_final, size_patch):
    image_result = np.full((size_final, size_final, 3), -1, dtype=np.int32)
    Ismp = np.array(Ismp)
    random.seed()
    new_patch_random = random_patch(Ismp, size_patch)
    # Trouver le milieu de image_vide
    milieu = (int(size_final/2), int(size_final/2))
    # Coller le patch au milieu de image_vide en gardant les contours inchangés
    image_result[milieu[0] - int(size_patch/2):milieu[0] + int(size_patch/2), milieu[1] - int(size_patch/2):milieu[1] + int(size_patch/2)] = new_patch_random
    return image_result

#Efros-Leung :
# Ismp : Image de texture qu' on utilise
# size_final : taille finale de l' image générée
# size_patch : la taille des patch que l'on utilisera (reste impair)
# epsilon : coef pour definir a quel point l'echantillon observé peut varier de son sample
def efros_leung(Ismp, size_final, size_patch, epsilon, image_result):
    #image_result = np.uint8(np.full((size_final, size_final,3),-1))
    #image_result = np.full((size_final, size_final, 3), -1, dtype=np.int32)
    Ismp = np.array(Ismp)
    #random.seed()
    #new_patch_random = random_patch(Ismp, size_patch)
    # Trouver le milieu de image_vide
    #milieu = (int(size_final/2), int(size_final/2))
    # Coller le patch au milieu de image_vide en gardant les contours inchangés
    #image_result[milieu[0] - int(size_patch/2):milieu[0] + int(size_patch/2), milieu[1] - int(size_patch/2):milieu[1] + int(size_patch/2)] = new_patch_random
    #Trouver le pixel de image_result qui possède le plus de voisins non vides
    #Pour chaque pixel de image_result, on regarde si il est vide ou non
    max_voisins = -1
    max_i = -1
    max_j = -1
    for i in range(size_final):
        for j in range(size_final):
            if image_result[i,j,0] == -1:
                #On regarde si il y a des voisins non vides
                voisins = []
                for k in range(i-1,i+1):
                    for l in range(j-1, j+1):
                        if k >= 0 and k < size_final and l >= 0 and l < size_final:
                            if image_result[k,l,0] != -1 :
                                voisins.append((k,l))
                #On choisit un voisin au hasard
                if len(voisins) > max_voisins:
                    max_voisins = len(voisins)
                    max_i = i
                    max_j = j
                    #print(max_i, max_j)
                '''if len(voisins) != 0:
                    random_voisin = random.choice(voisins)
                    #On trouve les patches similaires au patch du voisin
                    patches_similaires = []
                    for k in range(Ismp.shape[0] - size_patch):
                        for l in range(Ismp.shape[1] - size_patch):
                            if patch_similarity(Ismp[k:k+size_patch,l:l+size_patch], image_result[random_voisin[0]-int(size_patch/2):random_voisin[0]+int(size_patch/2),random_voisin[1]-int(size_patch/2):random_voisin[1]+int(size_patch/2)], epsilon):
                                patches_similaires.append((k,l))
                    #On choisit un patch similaire au hasard
                    if len(patches_similaires) != 0:
                        random_patch_similaire = random.choice(patches_similaires)
                        image_result[i,j] = Ismp[random_patch_similaire[0],random_patch_similaire[1]]'''
    #print("Indices ij pixel max voisin : ", max_i, ", ", max_j)            
    patch_pixel = extract_patch(image_result, size_patch, max_i, max_j)
    # Parcourir tous les pixels de l'image de textures pour comparer avec le patch_pixel
    offset = int((size_patch - 1)/2)
    omegap = []
    #print("Taille de la texture", Ismp.shape)
    for o in range(offset, Ismp.shape[0] - offset):
        for p in range(offset, Ismp.shape[1] - offset):
            patch_compare = extract_patch(Ismp, size_patch, o, p)
            if patch_similarity(patch_pixel, patch_compare, epsilon):
                #print("Patch similaire trouvé")
                omegap.append(patch_compare)
    if omegap != []:
        random_omegap = random.choice(omegap)
        #image_result[max_i - offset:max_i + offset, max_j - offset:max_j + offset] = random_omegap
        image_result[max_j, max_i] = random_omegap[offset + 1, offset + 1]
    return image_result


final = image_initiale(text0, 100, 10)
final = efros_leung(text0, 100, 10, 15, final)
for i in range (0, 10):
    #probleme : y'a que 3 pixels
    final = efros_leung(text0, 100, 10, 15, final)
final = Image.fromarray(final.astype('uint8'))
final.show()

# 1) On prend un patch au hasard dans l'image
# 2) On cherche les patches similaires dans l'image
# 3) On en prend un au hasard parmi les patches similaires
# 4) On le colle sur l'image finale
# 5) On recommence jusqu'à remplir l'image finale

# %%
