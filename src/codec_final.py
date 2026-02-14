import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

P = np.zeros((8, 8))
for k in range(8):
    for i in range(8):
        C_k = np.sqrt(1 / 2) if k == 0 else 1
        P[k, i] = np.sqrt(2 / 8) * C_k * np.cos((np.pi / (2 * 8)) * (2 * i + 1) * k)

def dct2(array):
    #Formule de DCT-II en calculant directement D (sans passer par P)
    #Cela fonctionne mais est très lent, et en plus n'est pas optimal pour décompresser
    D = np.zeros_like(array)
    C = [1/math.sqrt(2), 1, 1, 1, 1, 1, 1, 1]
    for k in range(0, 8):
        for l in range(0, 8):
            somme = 0
            #Création de la double somme
            for i in range(8):
                for j in range(8):
                    somme += array[i, j]*math.cos(((2*i+1)*k*math.pi)/16)*math.cos(((2*j+1)*l*math.pi)/16)

            D[k, l] = (1/4)*C[k]*C[l]*somme

    return D

def bruit(array, n):
    M = np.reshape(array, (8, 8))
    mask = np.fromfunction(lambda i, j: (i + j) > n, M.shape, dtype=int)
    M[mask] = 0
    return M


def tronc_arf(ar):
    x = np.shape(ar)[0]
    y = np.shape(ar)[1]
    x_coef = x//8
    y_coef = y//8
    x = 8*(x_coef)
    y = 8*(y_coef)
    tronc_ar = ar[0:x, 0:y, :]   
    return tronc_ar

def compressU(ar, n=8):
    x = np.shape(ar)[0]
    y = np.shape(ar)[1]
    x_coef = x//8
    y_coef = y//8
    x = 8*(x_coef)
    y = 8*(y_coef)
    tronc_ar = ar[0:x, 0:y] #Tronquer l'image par des multiples de 8
    tronc_ar = tronc_ar - 128*np.ones((x, y)) #Mettre les valeurs de -128 à 127
    new_ar = np.zeros_like(tronc_ar)
    plt.imshow(tronc_ar)
    plt.show()

    for ind1 in range(0, x_coef): #Double boucle pour parcourir chaque bloc
        for ind2 in range(0, y_coef):
            ar_inter = tronc_ar[ind1*8:(ind1+1)*8, ind2*8:(ind2+1)*8] #Bloc d'image de 8*8
            #D = dct2(ar_inter)
            D = P@ar_inter@P.transpose()
            new_ar[ind1*8:(ind1+1)*8, ind2*8:(ind2+1)*8] = bruit(D/Q,n)
            
    tauxC = np.count_nonzero(new_ar) #Compte le nb de valeurs non nulles
    return new_ar, tauxC #+ 128*np.ones(x,y)

def compress(imgLink, n=8):
    tauxCFinal = 0
    img = Image.open(imgLink)
    plt.imshow(img)
    plt.show()
    ar = np.array(img)
    ar = tronc_arf(ar) #Tronquer l'image par des multiples de 8
    plt.imshow(ar)
    plt.show()
    plt.imshow(ar-128)
    plt.show()
    comp=np.zeros(np.shape(ar))
    for i in range(np.shape(ar)[2]): #Pour chaque couleur/transparence
        res, tauxC = compressU(ar[:, :, i], n) #Transformer la couleur en compressé
        comp[:, :, i] = res #Le rajouter à l'image finale
        tauxCFinal += tauxC
    plt.imshow(comp)
    plt.show()
    tauxCFinal = tauxCFinal/((np.count_nonzero(ar))*3)
    return comp, tauxCFinal


def decompressU(ar):
    x = np.shape(ar)[0]
    y = np.shape(ar)[1]
    x_coef = x//8
    y_coef = y//8
    x = 8*(x_coef)
    y = 8*(y_coef)
    tronc_ar = ar[0:x, 0:y] #Tronquer l'image par des multiples de 8
    final_array = np.zeros_like(tronc_ar)

    for ind1 in range(0, x_coef): #Double boucle pour parcourir chaque bloc
        for ind2 in range(0, y_coef):
            ar_inter = tronc_ar[ind1*8:(ind1+1)*8, ind2*8:(ind2+1)*8] #Bloc d'image de 8*8
            D_tilde = ar_inter*Q
            M = P.transpose()@D_tilde@P
            final_array[ind1*8:(ind1+1)*8, ind2*8:(ind2+1)*8] = M

    return final_array #+ 128*np.ones((x, y))

def decompress(ar, neg=0):
    decomp = np.zeros(np.shape(ar))
    for i in range(np.shape(ar)[2]):
        decomp[:, :, i] = decompressU(ar[:, :, i])
    if neg:
        return 1-decomp
    return decomp




imgLink = "../images/earth.png"
img = Image.open(imgLink)
ar = np.array(img)
new_ar, taux_de_compression = compress(imgLink, n=6)
final = decompress(new_ar, neg=0)
final = (final + np.ones_like(final)*128)/255
plt.imshow(final)
plt.text(0, 0, f"Le taux de compression est de {round((1-taux_de_compression)*100, 1)}%")
plt.show()


### Code pour L'erreur en fonction des frequences
### et le taux de compreesion en fonction des frequences
'''
frequencies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
imgLink = "../images/earth.png"
img = Image.open(imgLink)
ar = np.array(img)

tauxErr = []
tauxComp = []

for i in range(len(frequencies)):
    new_ar, taux_de_compression = compress(imgLink, n=frequencies[i])
    tauxComp.append(taux_de_compression)
    final = decompress(new_ar, neg=0)
    final = (final + np.ones_like(final)*128)/255
    err = np.linalg.norm(ar-final)
    tauxErr.append(err)

plt.plot(frequencies, tauxComp, marker='o', label="Compression" , linestyle='-', color="red")
plt.xlabel("Fréquence")
plt.ylabel("Taux de Compression")
plt.title("Taux de Compression en fonction de la Fréquence")
plt.legend()
plt.show()

plt.plot(frequencies, tauxErr, marker='o', label="Erreur" , linestyle='-', color="blue")
plt.xlabel("Fréquence")
plt.ylabel("Erreur")
plt.title("Erreur en fonction de la Fréquence")
plt.legend()
plt.show()
'''