from projet_mfreq_jd_part1 import *

taille_im = 60
taille_patch_original = 16
taille_patch = 14 
mask = np.full((taille_im, taille_im), 0, dtype=np.uint8)
nv = np.full((taille_im, taille_im), 0, dtype=np.uint8)
patches = []

final = image_initiale(text0, taille_im, taille_patch, taille_patch_original)
for i in range (0, (taille_im*taille_im) - (taille_patch*taille_patch)):
    final = efros_leung(text0, taille_im, taille_patch, 0.0, final)
final = Image.fromarray(final.astype('uint8'))
final.save("texture_generee.png")
final.show()

# Enregistrez le temps de fin
end_time = time.time()

# Calculez la différence pour obtenir le temps écoulé
elapsed_time = end_time - start_time

# Affichez le temps écoulé
print(f"Temps écoulé pour une image de {taille_im}x{taille_im} : {elapsed_time} secondes")

