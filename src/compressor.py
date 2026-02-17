import numpy as np

class JPEGCompressor:
    def __init__(self, quantization_matrix):
        """
        On initialise le compresseur avec une matrice Q spécifique.
        """
        self.Q = quantization_matrix
        self.P = self._generate_dct_matrix(8)

    def _generate_dct_matrix(self, n=8):
        """Méthode privée pour générer la matrice de passage P."""
        P = np.zeros((n, n))
        for k in range(n):
            for i in range(n):
                ck = 1/np.sqrt(2) if k == 0 else 1
                P[k, i] = np.sqrt(1/4) * ck * np.cos(((2*i + 1)*k*np.pi)/16)
        return P

    def compress_block(self, block):
        """Applique DCT + Quantification sur un bloc 8x8."""
        # On passe en fréquentiel
        dct_coefs = self.P @ block @ self.P.T
        # Quantification avec l'arrondi (np.round) qu'on a vu ensemble
        quantized = np.round(dct_coefs / self.Q)
        return quantized

    def decompress_block(self, quantized_block):
        """Applique Déquantification + IDCT sur un bloc 8x8."""
        # On remultiplie par Q
        dct_recons = quantized_block * self.Q
        # Retour dans l'espace des pixels
        block_recons = self.P.T @ dct_recons @ self.P
        return block_recons

    def process_image(self, image, mode='compress'):
        """
        Gère le découpage en blocs de toute l'image.
        Accepte des images 2D (gris) ou 3D (RGB).
        """
        # On s'assure que l'image est en float pour les calculs
        img = image.astype(np.float64)
        h, w = img.shape[:2]
        result = np.zeros_like(img)
        
        # Gestion des canaux (1 ou 3)
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        
        for c in range(channels):
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    # Extraction du bloc
                    if channels == 1:
                        block = img[i:i+8, j:j+8]
                    else:
                        block = img[i:i+8, j:j+8, c]
                    
                    # Traitement
                    if mode == 'compress':
                        res_block = self.compress_block(block)
                    else:
                        res_block = self.decompress_block(block)
                        
                    # Insertion dans le résultat
                    if channels == 1:
                        result[i:i+8, j:j+8] = res_block
                    else:
                        result[i:i+8, j:j+8, c] = res_block
        return result
    
class ImageMetrics:
    @staticmethod
    def l2_relative_error(original, reconstructed):
        # On ramène tout dans le même référentiel (-128 à 127)
        diff = original.astype(float) - reconstructed.astype(float)
        return np.linalg.norm(diff) / np.linalg.norm(original)

    @staticmethod
    def compression_rate(quantized_image):
        total_pixels = quantized_image.size
        non_zero = np.count_nonzero(quantized_image)
        return 1.0 - (non_zero / total_pixels)