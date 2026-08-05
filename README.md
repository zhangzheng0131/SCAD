
# SPADE: Detection of Spatially Aberrant Cells in Spatial Transcriptomics Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-GPB-red.svg)](https://www.sciencedirect.com/journal/genomics-proteomics-and-bioinformatics)


![Pipeline](pipeline.png)

**SCAD** is a computational framework designed to integrate single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (ST) data to quantitatively characterize and detect spatially aberrant cells. SCAD integrates scRNA-seq and spatial transcriptomics (ST) data through two independent autoencoders to generate latent embeddings for cells and spatial spots. Cell embeddings are modeled as cluster-aware latent representations using a Gaussian mixture model (GMM), where each component represents a distinct cell-type embedding. Each spot embedding is then decomposed into a weighted combination of GMM component centers, reflecting its underlying cellular composition. 
SCAD further learns a spatial mapping from ST embeddings to their physical coordinates, enabling reconstruction of tissue architecture. Spatially aberrant spots are subsequently identified as outliers in this mapping using an uncertainty quantification framework.
1.  **Joint Embedding:** Mapping scRNA-seq cells and ST spots into a shared latent space using a VAE.
2.  **Deconvolution:** Modeling ST spots as a weighted combination of cell-type embeddings (GMM components) to mitigate batch effects.
3.  **Spatial Mapping:** Learning a mapping from latent embeddings to physical coordinates.
4.  **Uncertainty Quantification:** Utilizing **Conformal Prediction** to generate statistically rigorous prediction intervals, identifying spots whose true location deviates significantly from their predicted location.

## ✨ Key Features

*   **Uncertainty-Calibrated Detection:** Unlike methods relying on arbitrary thresholds, SPADE uses conformal prediction to provide statistically principled detection with controlled false discovery rates.
*   **Integration:** Seamlessly integrates scRNA-seq and Spatial Transcriptomics data.
*   **Deconvolution:** Performs spatial deconvolution to resolve cell-type composition within spots.
*   **Robustness:** Demonstrated superior performance in identifying biologically meaningful aberrant spots in both simulated and real-world datasets (e.g., Human Squamous Cell Carcinoma).

## 🛠️ Installation

To run SPADE, you will need Python (3.8+) and the following dependencies:
- Python ≥ 3.8
- PyTorch ≥ 1.10
- Scanpy, anndata, pandas, numpy, scikit‑learn, scipy
- Other dependencies: tqdm, POT (for optimal transport), matplotlib, seaborn

### Install from source
```bash
git clone https://github.com/sohu-123/SPADE.git
cd SPADE
pip install -r requirements.txt
```
---

## 🚀 Quick Start

A complete example script `Example-SPADE.py` is provided in the repository. It demonstrates the full pipeline on PDAC (pancreatic ductal adenocarcinoma) data.

### 1. Prepare your data
- **scRNA‑seq**: AnnData object (cells × genes) with `.X` as expression matrix.
- **ST**: AnnData object (spots × genes) with `.obsm['spatial']` as 2D coordinates.
- **SVG list**: A list of spatially variable genes (e.g., from SpaGCN) to guide feature selection.

### 2. Configure paths
Edit the configuration section in `Example-SPADE.py`:
```python
INPUT_ADATA_ST = "/path/to/your/enhanced_ST.h5ad"
INPUT_ADATA_SC = "/path/to/your/scRNA.h5ad"
INPUT_SVG_LIST = "/path/to/svg_list.csv"
OUTPUT_BASE = "result/PDAC_final"
```

### 3. Run the pipeline
```bash
python Example-SPADE.py
```

This will:
- Preprocess and embed both datasets
- Train the VAE‑GMM model (3‑fold cross‑validation)
- Predict spatial coordinates for each spot
- Apply conformal prediction to detect aberrant spots
- Save results (predicted coordinates, aberrant labels, deconvolution scores, latent embeddings) in `OUTPUT_BASE/results/`

---

## 📁 Output Files

After running, the following files are created in the output directory:

| File | Description |
|------|-------------|
| `adata_total.h5ad` | Combined AnnData with latent embeddings and predicted spatial coordinates |
| `adata_sc.h5ad` / `adata_ST.h5ad` | Normalized single‑cell and ST data |
| `adata_sc_keep.h5ad` | Filtered scRNA‑seq cells that localize within tissue (resolution="low") |
| `trans_plan.csv` | Transport plan (if OT enabled) between cells and spots |
| `cluster_score.csv` | Cell‑type contribution scores per spot |
| `latent.csv` | Latent embeddings with batch labels |
| `spot_info_PDAC_final.txt` | Spot‑level metadata including aberrant flag |
| `eval_out_final.npz` | Coordinates, embeddings, and predictions for each cross‑validation fold |

---

## 📊 Usage Example (Code Snippet)

```python
import scanpy as sc
import numpy as np
from SPADE import Model3, conformal_prediction

# Load data
adata_sc = sc.read("scRNA.h5ad")
adata_st = sc.read("ST.h5ad")
svg_list = pd.read_csv("svg.csv", index_col=0).index

# Initialize model
model = Model3(
    resolution="low",
    batch_size=200,
    train_epoch=3000,
    sf_coord=50,
    rad_cutoff=1.2,
    seed=1234,
    lambdacos=10,
    lambdaSWD=5,
    lambdalat=10,
    device="cpu"
)

# Preprocess and train
K, clusters = model.preprocess(svg_list, adata_sc, adata_st)
model.train(training_idx_rna, training_idx_st)

# Evaluate and get embeddings
mu, phi, sigma, z_A, z_B, m_A, m_B = model.eval2()

# Detect aberrant spots
aberrant, confidence, lambda_calib, pred_coords, true_coords = conformal_prediction(
    true_coords=adata_st.obsm['spatial'],
    z_B=z_B,
    m_B=m_B,
    calib_index=val_idx,
    test_index=test_idx,
    alpha=0.05,
    k_neighbors=15
)
```

---

## 📈 Performance Highlights

- **Simulated mouse brain**: AUC = 0.991, AUPR = 8.1, recall = 90.5% with FPR < 5%.
- **Human SCC**: Identified 22 aberrant spots enriched in tumor‑specific keratin (TSK) regions and domains with high spatial variability. Detected genes (e.g., *EIF2AK1*, *TMSB15B*) associated with stress response and migration.
- **Xenium breast cancer**: Detected 6,544 aberrant cells enriched at invasive fronts and DCIS regions, capturing biologically meaningful heterogeneity.
- **State‑of‑the‑art comparison**: SPADE outperforms STALocator and scSpace in spatial mapping accuracy across multiple thresholds.

---

## 🧪 Customization

### Adjusting conformal prediction
- `alpha` – controls the nominal false discovery rate (default 0.05).
- `k_neighbors` – number of neighbors for local variability estimation (default 15).
- Increase `alpha` for stricter detection; decrease for more conservative.

### Model hyperparameters
Key parameters in `Model3`:
- `resolution` – `"low"` (coarse mapping) or `"high"` (enhanced single‑cell resolution, requires optimal transport).
- `train_epoch` – training epochs (default 3000; increase for larger datasets).
- `lambdaGAN`, `lambdacos`, `lambdaAE`, `lambdaLA`, `lambdaSWD` – loss weights for different objectives.

---


## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

**Happy spatial anomaly hunting!** 🧬🔬

