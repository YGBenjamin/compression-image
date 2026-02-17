# Image Compression & Signal Processing using DCT

This project implements an end-to-end image compression pipeline based on the **Discrete Cosine Transform (DCT)**. It bridges the gap between theoretical signal processing (Hilbert spaces) and practical Computer Vision.

## 🚀 Overview & Interactive App
The repository features an interactive **Streamlit dashboard** that allows for real-time experimentation with compression parameters:
* **Interactive Comparison:** Side-by-side view with a slider to inspect compression artifacts.
* **Dynamic Quality Control:** Adjust the quantization factor to see the impact on file integrity.
* **Live Metrics:** Real-time calculation of the **Compression Rate** and **Relative L2 Error (RMSE)**.
* **Export:** Download the processed image directly from the interface.

## 🔬 Scientific Experiments (Notebook)
Beyond the web application, the Jupyter Notebook (`compression.ipynb`) documents more advanced research:
* **Robustness to Noise:** Testing the algorithm's performance on noisy signals.
* **Variable Block Sizes:** Analyzing how changing the standard 8x8 grid affects the "blocking effect."
* **Thresholding:** Manual frequency filtering by zeroing out specific DCT coefficients.

---

## 📐 Mathematical Foundations

The core of this project lies in the representation of an image as a function within a specific mathematical framework.

### 1. The Image as an $L^2$ Function
An image can be modeled as a function $f(x, y)$ of spatial coordinates. We consider this function to belong to the **$L^2$ space** (the space of square-integrable functions). This is crucial because $L^2$ is a **Hilbert space**, which provides us with:
* **A Scalar Product:** Allowing us to define the "closeness" of two images.
* **A Norm (Energy):** The $L^2$ norm $\|f\|_2 = \sqrt{\iint |f(x,y)|^2 dxdy}$ represents the total energy of the image signal.

### 2. The Discrete Cosine Transform (DCT)
The compression works by performing a **change of basis**. We move from the spatial domain (pixels) to the frequency domain using the DCT. 
* **Basis Functions:** We project the image onto a set of oscillating cosine functions of increasing frequencies.
* **Energy Compaction:** Natural images are "smooth," meaning their energy is concentrated in the low-frequency coefficients. In the DCT domain, most high-frequency coefficients are near zero.



### 3. Quantization and Information Loss
The compression occurs during the **Quantization** step. We divide the DCT coefficients by a psycho-visual matrix $Q$:
$$D_{quantized} = \text{round}\left( \frac{P M P^T}{Q} \right)$$
This step intentionally discards high-frequency information that the human eye cannot perceive. Because of **Parseval's Theorem**, the energy lost during this step in the frequency domain is exactly equal to the reconstruction error in the spatial domain.

### 4. Error Measurement: Relative $L^2$ Norm
To scientifically quantify the quality of the reconstruction ($\hat{f}$) compared to the original ($f$), we use the **Relative $L^2$ Error**:
$$\epsilon = \frac{\|f - \hat{f}\|_2}{\|f\|_2}$$
This metric is more robust than simple absolute error because it measures the percentage of energy lost, regardless of the image's overall brightness.

---

## 🛠️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/JPEG-Compression-DCT.git](https://github.com/YourUsername/JPEG-Compression-DCT.git)
   ```
