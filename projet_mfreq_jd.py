import numpy as np

from PIL import Image

text0 = Image.open("textures_data/text0.png")
text1 = Image.open("textures_data/text1.png")
text2 = Image.open("textures_data/text2.png")
text3 = Image.open("textures_data/text3.png")
text4 = Image.open("textures_data/text4.png")
text5 = Image.open("textures_data/text5.png")
text6 = Image.open("textures_data/text6.png")

#text0.show()
#text1bis = np.array(text1)
#print(text1bis.shape)

# faire une fonction qui prenne en entree une image couleur ou grayscale
# donc tester le nombre de channel et traiter l'image en fonction
# Paramètres : (Ismp, size_final, taille_patch (doit etre impair), epsilon)

def random_patch(Ismp, size_patch):
    Ismp = np.array(Ismp)
    patch = np.zeros((size_patch, size_patch))
    random_x = np.random.randint(0, Ismp.shape[0] - size_patch)
    random_y = np.random.randint(0, Ismp.shape[1] - size_patch)
    patch = Ismp[random_x:random_x + size_patch, random_y:random_y + size_patch]
    return patch

#Efros-Leung :
# Ismp : Image de texture qu' on utilise
# size_final : taille finale de l' image générée
# size_patch : la taille des patch que l'on utilisera (reste impair)
# epsilon : coef pour definir a quel point l'echantillon observé peut varier de son sample
def efros_leung(Ismp, size_final, size_patch, epsilon):
    image_result = np.uint8(np.full((size_final, size_final,3),-1))
    new_patch_random = random_patch(Ismp, size_patch)
    # Trouver le milieu de image_vide
    milieu = (int(size_final/2), int(size_final/2))
    # Coller le patch au milieu de image_vide en gardant les contours inchangés
    image_result[milieu[0] - int(size_patch/2):milieu[0] + int(size_patch/2), milieu[1] - int(size_patch/2):milieu[1] + int(size_patch/2)] = new_patch_random
    #Trouver le pixel de image_result qui possède le plus de voisins non vides
    
                

                        
    


    # On prend un patch au hasard dans l'image
    # On cherche les patches similaires dans l'image
    # On en prend un au hasard parmi les patches similaires
    # On le colle sur l'image finale
    # On recommence jusqu'à remplir l'image finale
    
final = efros_leung(text0, 100, 10, 0.1)
final = Image.fromarray(final)
final.show()

# 1) On prend un patch au hasard dans l'image
# 2) On cherche les patches similaires dans l'image
# 3) On en prend un au hasard parmi les patches similaires
# 4) On le colle sur l'image finale
# 5) On recommence jusqu'à remplir l'image finale
