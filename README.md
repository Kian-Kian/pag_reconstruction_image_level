# pag_reconstruction_image_level

Multi-branch image-level reconstruction of prior austenite grain IPF maps from martensitic EBSD images.

The single model uses alpha-phase IPF images. Dual and triple models use early-fused IPF and band contrast (BC), together with an auxiliary BC encoder. The triple model additionally uses a gamma-phase grain boundary (GB) encoder during training. GB inputs are randomly masked per sample; validation and normal inference bypass this encoder and use zero-filled GB bottleneck features.

## Environment

Python 3.12 and an NVIDIA GPU are required for training. Full-resolution attention can require substantial CPU memory; GPU evaluation is recommended. Evaluation selects CUDA when available and otherwise uses CPU; --device cpu or --device cuda selects a device explicitly. The experiments used PyTorch 2.8.0 with CUDA 12.8, NumPy 2.3.2, and Pillow 11.3.0. Install a PyTorch build compatible with the local GPU and driver, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Run all commands from the repository root. No pretrained backbone is used. Trained checkpoints and experiment outputs are not included.

## Data

The dataset directory contains the 100 original regions, with 400 PNG files in total:

```text
dataset/
  alpha/IPF/001.png ... 100.png
  alpha/BC/001.png ... 100.png
  gamma/IPF/001.png ... 100.png
  gamma/GB/001.png ... 100.png
```

Gamma-phase IPF and GB images are software-reconstructed references. Image-level color agreement does not directly measure crystallographic misorientation or grain boundary topology. Keep filenames, spatial alignment, and the original IPF color convention unchanged.

Prepare the fixed split and augmented data:

```bash
python scripts/prepare_data.py
```

Ten original regions are reserved for independent testing before augmentation. The other 90 regions produce 900 samples, split into 810 training and 90 validation samples. Different augmentations of a source region can occur in both training and validation; the independent test regions are excluded from that pool. Intensities are scaled to [0, 1].

## Train

```bash
python -m src.train --config configs/final_triple_seed312.json
python -m src.train --config configs/final_triple_seed312.json --resume
```

Existing runs cannot be silently restarted. Outputs are stored in outputs/<run_name>/. best.pt is selected by minimum validation composite loss; last.pt includes the optimizer and random states for resumption.

Final primary comparisons use base64, AdamW, learning rate 0.0002, weight decay 0.01, batch size 16, 200 epochs, BF16, and seeds 312/313/314. Triple training uses GB drop probability 0.7. The objective is L1 + (1 - SSIM) + cyclic hue distance + (1 - RGB cosine similarity), with all weights equal to one. SSIM uses a 3 x 3 uniform window.

## Evaluate

After training, evaluate the independent test set:

```bash
python -m src.evaluate --run final_triple_seed312
python -m src.evaluate --run final_triple_seed312 --save-cam --output outputs/final_triple_seed312/test_cam
```

Evaluation saves prediction and reference PNGs, mean absolute RGB error maps and floating-point error arrays, per-image metrics.csv, and summary.json. Metrics are computed at native resolution without per-image min-max normalization. Test metrics are averaged over regions; seed SD is the sample SD of the three seed-level means, not the SD of 30 independent regions.

CAM targets mean predicted RGB intensity and is independently ReLU/max-normalized. It describes output sensitivity; colors do not compare absolute importance across maps. Evaluation CAM uses normal inference with GB disabled. Training snapshots may also contain explicitly labeled GB-enabled diagnostic CAM.

## External data

```text
external_data/condition_name/
  alpha/IPF/sample01.png
  alpha/BC/sample01.png
  gamma/IPF/sample01.png
```

Reference gamma/IPF is needed only for quantitative evaluation. Single-model prediction requires only alpha/IPF. BC is required for dual/triple models. Target GB is not required for normal triple-model evaluation or prediction. GB-only is a diagnostic model using target-side boundary information.

```bash
python -m src.evaluate --run final_triple_seed312 --external external_data/condition_name --output outputs/final_triple_seed312/condition_name
python -m src.evaluate --run final_triple_seed312 --external external_data/condition_name --predict-only --output outputs/final_triple_seed312/condition_name_predictions
```

Use a separate output directory for each dataset. A non-empty output directory is rejected unless --overwrite is explicitly supplied. Prediction-only mode saves predictions without reference-based metrics.

## Experiment configurations

The repository includes 39 formal configurations: nine capacity/learning-rate/weight-decay runs, eight loss combinations, eleven GB drop probabilities, nine primary three-seed runs, and two single-modality diagnostic runs. Capacity uses FP16; other formal runs use BF16. capacity_48 is reused as the learning-rate 0.0002 and weight-decay 0.0001 baseline, so those comparisons include a precision difference. Sensitivity settings are selected independently by validation loss, not test performance.

```bash
python scripts/run_experiments.py capacity_16 capacity_32
python scripts/pipeline.py
python scripts/export_plot_data.py
```

The full pipeline trains the sensitivity experiments, selects final settings from validation history, trains final configurations, evaluates checkpoints, and exports plot data. Existing incompatible final configurations are protected against overwrite. The prepared final configurations reproduce the recorded experimental settings; reruns can produce different selected settings. Development checks, historical runs, prepared augmentations, external samples, and publication figures are not included.
