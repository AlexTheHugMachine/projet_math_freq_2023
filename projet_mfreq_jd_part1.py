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

taille_im = 60
taille_patch = 10
mask = np.full((taille_im, taille_im), 0, dtype=np.uint8)
nv = np.full((taille_im, taille_im), 0, dtype=np.uint8)

def ssd_pixel(pixel1, pixel2):
    difference = pixel1 - pixel2
    squared_difference = difference ** 2
    return squared_difference

# faire une fonction qui prenne en entree une image couleur ou grayscale
# donc tester le nombre de channel et traiter l'image en fonction
# Paramètres : (Ismp, size_final, taille_patch (doit etre impair), epsilon)
# Définir une fonction pour calculer la similarité entre deux patches

def patch_similarity(patch1, patch2):
    valid_indices = (patch1[..., 0] != -1) & (patch2[..., 0] != -1)
    
    valid_pixels_patch1 = patch1[valid_indices]
    valid_pixels_patch2 = patch2[valid_indices]
    
    if valid_pixels_patch1.size == 0:
        return 0.0
    
    diff_rgb = np.abs(valid_pixels_patch1 - valid_pixels_patch2)
    diff_sum = np.sum(diff_rgb, axis=1)
    
    return np.mean(diff_sum) / 3.0

    '''
    if diffR/nbpix < epsilon and diffG/nbpix < epsilon and diffB/nbpix < epsilon:
        return True
    else: return False'''

#ajouter une comparaison de distance différente (SSD) pour voir la différence

def random_patch(Ismp, size_patch):
    Ismp = np.array(Ismp)
    patch = np.zeros((size_patch, size_patch))
    random_x = np.random.randint(0, Ismp.shape[0] - size_patch)
    random_y = np.random.randint(0, Ismp.shape[1] - size_patch)
    patch = Ismp[random_x:random_x + size_patch, random_y:random_y + size_patch]
    return patch

def extract_patch(image, size, i_centre, j_centre):
    patch = np.full((size, size, 3), -1, dtype=np.int32)
    offset = size // 2
    
    i_min = max(0, i_centre - offset)
    i_max = min(image.shape[0], i_centre + offset)
    j_min = max(0, j_centre - offset)
    j_max = min(image.shape[1], j_centre + offset)
    
    patch_start_i = offset - min(i_centre, offset)
    patch_end_i = patch_start_i + (i_max - i_min)
    patch_start_j = offset - min(j_centre, offset)
    patch_end_j = patch_start_j + (j_max - j_min)
    
    patch[patch_start_j:patch_end_j, patch_start_i:patch_end_i] = image[j_min:j_max, i_min:i_max]
    
    return patch

def image_initiale(Ismp, size_final, size_patch):
    global mask
    global nv
    image_result = np.full((size_final, size_final, 3), -1, dtype=np.int32)
    Ismp = np.array(Ismp)
    random.seed()
    r = int(size_patch/2)
    new_patch_random = random_patch(Ismp, size_patch)
    # Trouver le milieu de image_vide
    milieu = (int(size_final/2), int(size_final/2))
    # Coller le patch au milieu de image_vide en gardant les contours inchangés
    image_result[milieu[1] - r:milieu[1] + r, milieu[0] - r:milieu[0] + r] = new_patch_random
    mask[milieu[1] - r:milieu[1] + r, milieu[0] - r:milieu[0] + r] = 1
    '''for i in range(milieu[0] - r, milieu[0] + r + 1) :
        for j in range(milieu[1] - r, milieu[1] + r + 1) :
            compte_voisins(j, i)'''
    nv[milieu[1]-r : milieu[1]+r+1 , milieu[0]-r : milieu[0]+r+1] += 1
    return image_result

#Efros-Leung :
# Ismp : Image de texture qu' on utilise
# size_final : taille finale de l' image générée
# size_patch : la taille des patch que l'on utilisera (reste impair)
# epsilon : coef pour definir a quel point l'echantillon observé peut varier de son sample
# mask : tableau de booleen de taille size_final x size_final
# neighbors : tableau de taille size_final x size_final qui contient le nb de voisins remplis du pixel
def efros_leung(Ismp, size_final, size_patch, epsilon, image_result):
    global mask
    global nv
    Ismp = np.array(Ismp)
    #random.seed()
    #new_patch_random = random_patch(Ismp, size_patch)
    # Trouver le milieu de image_vide
    #milieu = (int(size_final/2), int(size_final/2))
    # Coller le patch au milieu de image_vide en gardant les contours inchangés
    #image_result[milieu[0] - int(size_patch/2):milieu[0] + int(size_patch/2), milieu[1] - int(size_patch/2):milieu[1] + int(size_patch/2)] = new_patch_random
    #Trouver le pixel de image_result qui possède le plus de voisins non vides
    #Pour chaque pixel de image_result, on regarde si il est vide ou non

    #trouver pixel non rempli avec plus petite valeur
    indice_maximum_voisins = np.argmax(np.multiply((1 - mask), nv))
    max_i = np.unravel_index(indice_maximum_voisins, (size_final, size_final))[1]
    max_j = np.unravel_index(indice_maximum_voisins, (size_final, size_final))[0]

    '''max_voisins = -1
    max_i = -1
    max_j = -1
    for i in range(size_final):
        for j in range(size_final):
            if mask[j,i] == 0 :
                if nv[j,i] > max_voisins:
                    max_voisins = nv[j,i]
                    max_i = i
                    max_j = j'''

    
                
    #print("Indices ij pixel max voisin : ", max_i, ", ", max_j)  
    #print("Valeur pixel : ", image_result[max_j, max_i, 0], ", ", image_result[max_j, max_i, 0], ", ", image_result[max_j, max_i, 0])        
    patch_pixel = extract_patch(image_result, size_patch, max_i, max_j)
    # Parcourir tous les pixels de l'image de textures pour comparer avec le patch_pixel
    offset = int((size_patch - 1)/2)
    omegap = []
    #print("Taille de la texture", Ismp.shape)
    best_diff = 255
    best_patch = random_patch(Ismp, size_patch)
    for o in range(offset, Ismp.shape[0] - offset):
        for p in range(offset, Ismp.shape[1] - offset):
            patch_compare = extract_patch(Ismp, size_patch, o, p)
            similarity = patch_similarity(patch_pixel, patch_compare)
            if similarity <= epsilon:
                omegap.append(patch_compare)
            else :
                if similarity < best_diff :
                    best_diff = similarity
                    best_patch = patch_compare
            '''if patch_similarity(patch_pixel, patch_compare, epsilon):
                #print("Patch similaire trouvé")
                omegap.append(patch_compare)'''
    #print("taille omegap :", len(omegap))
    #print(nv[max_j, max_i])
    if omegap != []:
        random_omegap = random.choice(omegap)
        image_result[max_j, max_i] = random_omegap[offset + 1, offset + 1]
        mask[max_j, max_i] = 1
        if(max_i == 0 and max_j == 0):
            nv[max_j:max_j+offset+1, max_i:max_i+offset+1] += 1
        elif(max_i == 0):
            nv[max_j-offset:max_j+offset+1, max_i:max_i+offset+1] += 1
        elif(max_j == 0):
            nv[max_j:max_j+offset+1, max_i-offset:max_i+offset+1] += 1
        else:
        #compte_voisins(max_j, max_i)
            nv[max_j-offset : max_j+offset+1 , max_i-offset : max_i+offset+1] += 1
        #print("nv après : ", nv[max_j, max_i])
    else:
        #print("pas de match dans texture :  [", max_j, ", ", max_i, "]")
        #no_liste.append([max_j, max_i])
        image_result[max_j, max_i] = best_patch[offset + 1, offset + 1]
        mask[max_j, max_i] = 1
        #compte_voisins(max_j, max_i)
        nv[max_j-offset : max_j+offset+1 , max_i-offset : max_i+offset+1] += 1
    return image_result

final = image_initiale(text0, taille_im, taille_patch)
#final = efros_leung(text0, 20, 10, 15, final)
for i in range (0, (taille_im*taille_im) - (taille_patch*taille_patch)):
#for i in np.arange(0, (taille_im*taille_im) - (taille_patch*taille_patch), 1):
    #probleme : y'a que 3 pixels
    final = efros_leung(text0, taille_im, taille_patch, 10, final)
    #print("pixels ecrits : ", pixels_ecrits)
    #print("iteration : ", i)
final = Image.fromarray(final.astype('uint8'))
final.show()

# 1) On prend un patch au hasard dans l'image
# 2) On cherche les patches similaires dans l'image
# 3) On en prend un au hasard parmi les patches similaires
# 4) On le colle sur l'image finale
# 5) On recommence jusqu'à remplir l'image finale

# %%
