# Colab Run Guide

## 1) Open Colab and set runtime to GPU
Runtime -> Change runtime type -> T4/L4/A100

## 2) Clone and install
```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd emg-silent-speech-decoder
!pip install -r requirements.txt
!pip install -e .
```

## 3) Download study data (Zenodo)
```python
!python scripts/download_data.py --config configs/base.yaml --source zenodo_vss --record_id 4064408
```

## 4) Optional: import clean labeled subset for testing
```python
!python scripts/import_vss_labeled_subset.py --config configs/base.yaml --max_samples 1000 --prefix labeledvss
```

## 5) Train with resume + autosave
```python
!python -u scripts/train_phoneme_ctc.py --config configs/train_phoneme_ctc.yaml --max_steps 2000 --resume --save_every_steps 25 --log_every_steps 10 --metrics_file outputs/train_metrics.jsonl
```

You will see periodic logs with:
- `steps_per_sec`
- `elapsed_min`
- `eta_min`

The same values are appended to:
- `outputs/train_metrics.jsonl`

## 6) Save checkpoint to Drive (recommended)
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/emg_checkpoints
!cp checkpoints/phoneme_ctc.pt /content/drive/MyDrive/emg_checkpoints/
```

## 7) Download checkpoint to laptop
```python
from google.colab import files
files.download('checkpoints/phoneme_ctc.pt')
```

## 8) Test locally on laptop after copying checkpoint
```powershell
python scripts/infer_phoneme.py --config configs/infer.yaml --split test --num_samples 5 --decode beam
```
