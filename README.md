# Deep Gesture Recognition under Data Loss (DGRDL)

Official code for **Deep Gesture Recognition under Data Loss**, published as an Early Access article in *IEEE Sensors Journal*. The framework provides robust FMCW mmWave radar hand-gesture recognition under interference-induced data loss (DL).

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11595268)  
**DOI:** [10.1109/JSEN.2026.3708066](https://doi.org/10.1109/JSEN.2026.3708066)  
**Dataset:** [IEEE DataPort](https://dx.doi.org/10.21227/zkfq-ek87)

## Method overview

1. **BSMamba encoder–decoder** reconstructs clean latents from corrupted DRAI inputs (`20 × 32 × 32`).
2. **Scale-Shift MLP** applies class-conditioned affine transforms on the bottleneck latent and selects the most confident hypothesis at inference.
3. **Two-phase training**
   - Phase 1: MSE reconstruction on randomly zeroed (DL) inputs.
   - Phase 2: freeze encoder; train Scale-Shift + classifier with cross-entropy.


## Repository layout


Deep-Gesture-Recognition-Data-Loss/
├── configs/default.yaml
├── dgrdl/
│   ├── models/          # BSMamba encoder-decoder + Scale-Shift
│   └── data/            # dataset
├── scripts/
│   ├── train.py         # two-phase DGRDL training
│   └── evaluate.py
├── requirements.txt
└── README.md


## Installation


conda create -n dgrdl python=3.10 -y
conda activate dgrdl
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH}"


> **Note:** `mamba-ssm` typically requires a CUDA GPU build matching your PyTorch/CUDA versions.

## Data

1. Prepare the dataset.
2. Convert raw `.mat` recordings to DRAI `.npy` tensors (optional if already preprocessed):


python scripts/preprocess_drai.py \
  --input-dir /path/to/mat_files \
  --output-dir /path/to/npy_drai


3. Edit `configs/default.yaml` and set `data.root_dirs` (and optional `negative_root`) to your `.npy` folders.

Expected tensor shape per sample: **`(T, H, W)`** with `T≈20`, resized to `20 × 32 × 32` during loading.

Filename convention used by the loader: `*_GestureName.npy` (gesture token must match the class map in `dgrdl/data/dataset.py`).

## Training

### Proposed method (two-phase)


# Both phases
python scripts/train.py --config configs/default.yaml --dl-ratio 0.2 --phase all

# Or separately
python scripts/train.py --config configs/default.yaml --phase 1 --dl-ratio 0.2
python scripts/train.py --config configs/default.yaml --phase 2 --dl-ratio 0.2 \
  --resume-phase1 checkpoints/phase1_encoder_decoder.pth


Checkpoints:
- `checkpoints/phase1_encoder_decoder.pth`
- `checkpoints/best_model.pth`

## Evaluation


python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pth \
  --dl-ratios 0.0 0.2 0.4 0.6


## Citation

If you use this code or dataset, please cite the paper:


@article{kajbaf2026dgrdl,
  title   = {Deep Gesture Recognition under Data Loss},
  author  = {Kajbaf, Amin and Yazdian, Ehsan and Akhaee, Mohammad Ali and Toosi, Ramin and Gazor, Saeed},
  journal = {IEEE Sensors Journal},
  year    = {2026},
  doi     = {10.1109/JSEN.2026.3708066},
  note    = {Early Access}
}


## License

Code released for research use accompanying the paper. Please contact the authors for other licensing requests.
Email: A.kajbaf@ec.iut.ac.ir
