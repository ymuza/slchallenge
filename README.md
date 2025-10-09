# Deep K-Correct

This package contains utilities to calculate galaxy K corrections using a fine-tuned version of AstroCLIP foundation model.
- K corrections can be calculated either from galaxy images (in bands g, r and z) or from galaxy spectra.
- A statistical comparison is made against the SOTA tool "kCorrect", which is not ML based.

# Setting up the environment

Create a conda environment and install requirements.txt

```bash
conda create --name deep-k-correct python=3.9
conda activate deep-k-correct
pip install torch==2.0.0+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -e .
```
# Preprocess the data from the original datasets

For completion, these instructions explain how to reconstruct the K correction train/test dataset for Desi-EDR galaxies. Run the following steps from the root directory of the repository:

## Download AstroCLIP dataset

Approximately 200K galaxies from DESI-edr described by their images, spectra and redshift. 60 GB.

```python
python scripts/download_astroclip.py data/raw/AstroCLIP
```

## Download Fastspecfit VAC

Fastspecfit Value Added Catalog exports high-quality estimation of K corrections:

```bash
wget -P data/raw/ https://data.desi.lbl.gov/public/edr/vac/edr/fastspecfit/fuji/v3.2/catalogs/fastspec-fuji.fits
```

## Download DESI Legacy Fluxes

I made the relevant deredened fluxes at bands g,r and z avalible in kaggle:

```bash
kaggle astro_datasets download -d jeremiasrodriguez/flux-in-bands-g-r-and-z-for-galaxies-in-desi-edr -p data/
unzip data/flux-in-bands-g-r-and-z-for-galaxies-in-desi-edr.zip -d data/
```

Alternatively, you may recompute this dataset. The following script will query the Astro Data Lab database and retrieve the fluxes for matching target ids. This may take a while to run because it is O(n^2) on the size of the tables.

```python
python scripts/download_DESI_legacy_fluxes.py data/raw/AstroCLIP data/desi_edr_fluxes.h5
```

## Download pretrained model checkpoint

Download the checkpoint of the pretrained AstroCLIP foundation model:

```bash
wget -P data/checkpoints/ https://huggingface.co/polymathic-ai/astroclip/resolve/main/astroclip.ckpt 
```

# Comparison of Blanton K corrections and Fastspecfit K corrections

If you wish to recalculate Blanton K corrections from the datasets downloaded in the step before, run:

```python
python blanton_analysis/calculate_blanton_K_corrections.py --astroclip_path data/raw/AstroCLIP --desi_path  data/desi_edr_fluxes.h5 
```

Alternatively, you can download the pre-calculated K corrections:

```bash
kaggle astro_datasets download -d jeremiasrodriguez/blanton-k-corrections-for-astroclip-dataset -p data/
unzip data/blanton-k-corrections-for-astroclip-dataset.zip -d data/
```

For a statistical comparison between Blanton K corrections and Fastspecfit VAC K-corrections, check out this jupyter notebook:

`blanton_analysis/blanton_fastspecfit_analysis.ipynb`

# AstroCLIP dataset

The AstrCLIP dataset can be visualized via the following script:

`kcorrection/visualize_AstroCLIP.ipynb` [TESTED]

The original dataset has spectra and images. We need to load the pretrained foundation model and calculate embeddings in order to fine tune the model. We can do this by running:

```bash
python kcorrection/embed_astroclip.py --model_path data/checkpoints/astroclip.ckpt --dataset_path data/raw/AstroCLIP/ --loader_type train  data/train_embeddings.h5
python kcorrection/embed_astroclip.py --model_path data/checkpoints/astroclip.ckpt --dataset_path data/raw/AstroCLIP/ --loader_type val  data/val_embeddings.h5
```

Then create the K correction train-test dataset:

```bash
 python kcorrection/generate_k_correction_dataset.py data/train_embeddings.h5 data/val_embeddings.h5 data/raw/fastspec-fuji.fits data/
```

