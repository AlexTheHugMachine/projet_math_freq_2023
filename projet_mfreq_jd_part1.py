import numpy as np
import random
import time

from PIL import Image

text0 = Image.open("textures_data/text0.png").convert('RGB')
text1 = Image.open("textures_data/text1.png").convert('RGB')
text2 = Image.open("textures_data/text2.png").convert('RGB')
text3 = Image.open("textures_data/text3.png").convert('RGB')
text4 = Image.open("textures_data/text4.png").convert('RGB')
text5 = Image.open("textures_data/text5.png").convert('RGB')
text6 = Image.open("textures_data/text6.png").convert('RGB')

# Enregistrez le temps de début
start_time = time.time()

patches = []

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
        #return 0.0
        # on a aucun pixel comparable
        return 1.0
    
    diff_rgb = np.abs(valid_pixels_patch1 - valid_pixels_patch2)
    diff_sum = np.sum(diff_rgb, axis=1)
    
    # / 225 pour normaliser et avoir un epsilon entre 0 et 1
    return (np.mean(diff_sum) / 3.0) / 255


def ssd(A, B):
    return np.sum((A - B)**2)


def patch_similarity_ssd(patch1, patch2):
    # Les patchs doivent avoir la même taille pour que cela fonctionne
    valid_indices = (patch1[..., 0] != -1) & (patch2[..., 0] != -1)
    
    valid_pixels_patch1 = patch1[valid_indices]
    valid_pixels_patch2 = patch2[valid_indices]

    
    nb_values = np.prod(patch1.shape)

    sumR = ssd(valid_pixels_patch1[:,0], valid_pixels_patch2[:,0])
    sumG = ssd(valid_pixels_patch1[:,1], valid_pixels_patch2[:,1])
    sumB = ssd(valid_pixels_patch1[:,2], valid_pixels_patch2[:,2])

    epsilon = (sumR + sumG + sumB) / nb_values / (255 **2)
    return epsilon


def random_patch(Ismp, size_patch):
    Ismp = np.array(Ismp)
    patch = np.zeros((size_patch, size_patch))
    random_x = np.random.randint(0, Ismp.shape[0] - size_patch)
    random_y = np.random.randint(0, Ismp.shape[1] - size_patch)
    patch = Ismp[random_x:random_x + size_patch, random_y:random_y + size_patch]
    return patch


# Plus utile si on utilise la matrice de patches
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


def image_initiale(Ismp, size_final, size_patch, size_patch_origine):
    global mask
    global nv
    global patches
    patches = []
    mask = np.full((size_final, size_final), 0, dtype=np.uint8)
    nv = np.full((size_final, size_final), 0, dtype=np.uint8)
    image_result = np.full((size_final, size_final, 3), -1, dtype=np.int32)
    Ismp = np.array(Ismp)
    random.seed()
    r = int(size_patch_origine/2)
    new_patch_random = random_patch(Ismp, size_patch_origine)
    milieu = (int(size_final/2), int(size_final/2))

    # Coller le patch au milieu de image_vide en gardant les contours inchangés
    image_result[milieu[1] - r:milieu[1] + r, milieu[0] - r:milieu[0] + r] = new_patch_random
    mask[milieu[1] - r:milieu[1] + r, milieu[0] - r:milieu[0] + r] = 1
    nv[milieu[1]-r : milieu[1]+r+1 , milieu[0]-r : milieu[0]+r+1] += 1

    # On découpe l'image de texture en patch qu'on stocke dans un tableau de patches
    dimx = Ismp.shape[0]
    dimy = Ismp.shape[1]
    #for i in range(dimx - taille_patch + 1):
    for i in range(dimx - size_patch + 1):
        for j in range(dimy - size_patch + 1):
            patches.append(Ismp[j:j+size_patch, i:i+size_patch])
    patches = np.array(patches)
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

    # trouver pixel non rempli avec plus petite valeur
    indice_maximum_voisins = np.argmax(np.multiply((1 - mask), nv))
    max_i = np.unravel_index(indice_maximum_voisins, (size_final, size_final))[1]
    max_j = np.unravel_index(indice_maximum_voisins, (size_final, size_final))[0]

    # On veut traiter les cas où le pixel central est proche des bords : 
    #  on gère les conflits de taille de patchs.
    id_j_min = int(max_j - (size_patch / 2))
    id_j_max = int(max_j + (size_patch / 2))
    id_i_min = int(max_i - (size_patch / 2))
    id_i_max = int(max_i + (size_patch / 2))

    good_j_min = max(0, id_j_min)
    good_i_min = max(0, id_i_min)
    good_j_max = min(size_final, id_j_max)
    good_i_max = min(size_final, id_i_max)

    if id_j_min < 0 :
        offset_j_min = abs(id_j_min)
    else :
        offset_j_min = 0

    if id_i_min < 0 :
        offset_i_min = abs(id_i_min)
    else :
        offset_i_min = 0

    if id_j_max >= size_final :
        offset_j_max = size_final - id_j_max
    else :
        offset_j_max = 0

    if id_i_max >= size_final :
        offset_i_max = size_final - id_i_max
    else :
        offset_i_max = 0


    patch_pixel = np.full((size_patch, size_patch, 3), -1)
    patch_pixel[0+offset_j_min:size_patch+offset_j_max, 0+offset_i_min:size_patch+offset_i_max] = image_result[good_j_min:good_j_max, good_i_min:good_i_max]
    
    # Parcourir tous les pixels de l'image de textures pour comparer avec le patch_pixel
    omegap = []

    offset = int((size_patch - 1)/2)
    best_diff = 255
    best_epsilon = 1.0
    best_patch = random_patch(Ismp, size_patch)

    # On cherche tous les patchs qui respectent le threshold de + ou - epsilon en distance
    for a in range(patches.shape[0]):
        patch_compare = patches[a]
        similarity = patch_similarity_ssd(patch_pixel, patch_compare)
        if similarity <= epsilon: 
            omegap.append(patch_compare)
            if best_epsilon > similarity :
                best_epsilon = similarity
        else :
            if similarity < best_diff :
                best_diff = similarity
                best_patch = patch_compare

    # Si on a trouvé des matchs respectant epsilon : on en choisi un random.
    if omegap != []:
        random_omegap = random.choice(omegap)
        # On récupère le pixel du milieu du patch tiré au hasard pour donner
        #  la couleur à celui de l'image finale
        image_result[max_j, max_i] = random_omegap[offset + 1, offset + 1]
        mask[max_j, max_i] = 1
        if(max_i == 0 and max_j == 0):
            nv[max_j:max_j+offset+1, max_i:max_i+offset+1] += 1
        elif(max_i == 0):
            nv[max_j-offset:max_j+offset+1, max_i:max_i+offset+1] += 1
        elif(max_j == 0):
            nv[max_j:max_j+offset+1, max_i-offset:max_i+offset+1] += 1
        else:
            nv[max_j-offset : max_j+offset+1 , max_i-offset : max_i+offset+1] += 1
    # Sinon on choisi le meilleur trouvé        
    else:
        image_result[max_j, max_i] = best_patch[offset + 1, offset + 1]
        mask[max_j, max_i] = 1
        nv[max_j-offset : max_j+offset+1 , max_i-offset : max_i+offset+1] += 1

    # Dans tous les cas, on a mis à jour le masque des pixels et le tableau du nombre de voisins
    return image_result

def synthetise_texture(texture, taille_image, taille_patche, taille_patch_origine):
    print("taille texture : ", texture.shape)  
    final = image_initiale(texture, taille_image, taille_patche, taille_patch_origine)
    for i in range (0, (taille_image*taille_image) - (taille_patch_origine*taille_patch_origine)):
        #print("Pixel n°", i)
        final = efros_leung(texture, taille_image, taille_patche, 0.0, final)
    final = Image.fromarray(final.astype('uint8'))
    final.save("texture_generee.png")
    return final