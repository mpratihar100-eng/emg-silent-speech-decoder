# EMG Silent Speech Decoder

End-to-end baseline repository for decoding surface EMG (sEMG) into:
1. ARPAbet phoneme sequences (CTC)
2. Best-effort word-level text via lexicon-based wordification

Supports offline dataset training/evaluation and real-time streaming inference from serial (Raspberry Pi Pico style CSV lines) or simulated file stream.

## Assumptions
- Python 3.10+
- Internal training format is one `.npz` per utterance with keys:
  - `emg`: float array `[T, C]`
  - `sr`: scalar sampling rate
  - `text`: UTF-8 text (optional)
  - `phonemes`: ARPAbet sequence string (space-separated, optional)
  - `speaker_id`, `session_id`: strings (optional)
- If force-aligned phonemes are unavailable, synthetic placeholder phoneme labels are generated from text using a tiny fallback lexicon.
- Zenodo dataset download can fail in restricted networks; script supports synthetic fallback so pipeline stays runnable.

## Repository Layout
```text
emg-silent-speech-decoder/
  assets/
  configs/
  data/
  scripts/
  src/emg_ssd/
  tests/
```

## 10-minute Quickstart
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py --config configs/base.yaml --source zenodo_vss
python scripts/train_phoneme_ctc.py --config configs/train_phoneme_ctc.yaml --max_steps 20
python scripts/eval_phoneme_ctc.py --config configs/eval.yaml
python scripts/infer_phoneme.py --config configs/infer.yaml --split test --num_samples 3
python scripts/infer_stream.py --config configs/hardware_serial.yaml --simulate
```

## Laptop App (No Terminal Workflow)
You can run a local desktop app that wraps the pipeline as buttons.

```bash
python app_gui.py
```

or on Windows:
```bash
launch_app.bat
```

Inside the app:
- `Setup Environment`
- `Download Data`
- `Create Sample Manifest`
- `Import Sample Data`
- `Train (Smoke)`
- `Evaluate`
- `Infer Samples`
- `Stream Sim`

All command output is shown in the app log panel.

### Drag-and-Drop in the App
- Drop sample files/folders/archives directly into the app:
  - `.csv`, `.npy`, `.npz`
  - `.zip`, `.tar`, `.tar.gz`, `.tgz` (auto-extracted)
- Then click `Import Dropped Files`.
- The app stages files into `data/sample_raw/`, auto-writes `manifest.csv`, and runs sample import.
- Installed model/testing flows then use the same internal dataset (`data/internal/`).

### How Your Data Is Processed
1. File parse: each dropped file is converted to EMG matrix `[T, C]`.
2. Internal format: saved as `.npz` with keys `emg`, `sr`, `text`, `phonemes`, `speaker_id`, `session_id`.
3. Preprocessing (configurable in `configs/base.yaml`):
   - notch (50/60 Hz), bandpass (20-450 Hz), optional rectify/envelope, normalization.
4. Features:
   - raw framed vectors or STFT log-power.
5. Model:
   - CNN + BiLSTM (or Transformer) + CTC for phoneme decoding.
6. Word output:
   - baseline phoneme-to-word lexicon matching with edit-distance scoring.

### Visualizing Electrodes and Phoneme Mapping
Generate per-electrode plots and a phoneme timeline overlay:

```bash
python scripts/visualize_emg.py --config configs/infer.yaml --split test --index 0
```

Or with a specific internal sample:
```bash
python scripts/visualize_emg.py --config configs/infer.yaml --npz_path data/internal/test/sample_00000.npz
```

Outputs are written to `outputs/visuals/`:
- `*_emg.png`: raw vs preprocessed signal for each channel/electrode
- `*_phoneme_timeline.png`: frame-wise predicted phoneme segments over time

## Insert Your Own Sample Data
Use this flow to import your own quick test files and run inference immediately.

1) Create manifest template:
```bash
python scripts/prepare_sample_data.py --config configs/sample_import.yaml --create_template
```

2) Put your files in `data/sample_raw/` and edit `data/sample_raw/manifest.csv`.
- Supported files: `.csv`, `.npy`, `.npz`
- Expected shape per file: `[T, C]` where `C = data.expected_channels` (default 8)
- CSV can be either `ch0..chN` or `t,ch0..chN`

Manifest columns:
```text
file,sr,text,phonemes,speaker_id,session_id,split
```
- `phonemes` is optional for inference-only tests.
- `split` should be `train`, `val`, or `test`.

3) Import into internal format:
```bash
python scripts/prepare_sample_data.py --config configs/sample_import.yaml
```

4) Test your sample immediately:
```bash
python scripts/infer_phoneme.py --config configs/infer.yaml --split test --num_samples 5 --decode beam
```

Optional quick re-train including your imported train split:
```bash
python scripts/train_phoneme_ctc.py --config configs/train_phoneme_ctc.yaml --max_steps 50
```

## Common Commands
```bash
make setup
make download
make train-smoke
make eval
make infer
make stream-sim
make test
```

## Data Sources
- Voicing Silent Speech (official): https://github.com/dgaddy/silent_speech
- Zenodo dataset DOI: https://doi.org/10.5281/zenodo.4064408
- semg-asr example: https://github.com/MiscellaneousStuff/semg-asr
- EMG-UKA corpus paper: https://www.csl.uni-bremen.de/cms/images/documents/publications/WandJankeSchultz_IS14_EMG-UKA-Corpus.pdf
- EMG-UKA trial subset: https://www.kaggle.com/datasets/xabierdezuazo/emguka-trial-corpus
- Wand et al. 2009: https://www.isca-archive.org/interspeech_2009/wand09_interspeech.pdf
- Clinical proof-of-concept: https://pmc.ncbi.nlm.nih.gov/articles/PMC5851476/

## Preprocessing (Configurable)
- Notch: 50/60 Hz
- Bandpass: default 20-450 Hz
- Optional rectification and lowpass envelope
- Session z-score normalization (robust MAD option)
- Feature mode: STFT log-power per channel
- Raw mode: direct time-domain windows for 1D CNN

## Calibration Workflow
Fine-tune adapter/head with 2-5 minutes user data:
```bash
python scripts/calibrate_user.py --config configs/calibration.yaml --checkpoint checkpoints/phoneme_ctc.pt
```

## Streaming Mode
- Serial line format expected from Pico:
  `t,ch0,ch1,...,chN`
- If hardware unavailable: simulated stream from `.npz` file.

## Optional ElevenLabs TTS
```bash
set ELEVENLABS_API_KEY=your_key_here
python scripts/infer_phoneme.py --config configs/infer.yaml --tts
```
If key missing, code prints decoded text without crashing.

## Troubleshooting
- Excessive mains hum: enable notch, verify ground/reference electrode quality.
- Drift/saturation: check ADC gain and apply robust normalization.
- Electrode shift across sessions: run `calibrate_user.py` with fresh user samples.
- Poor wordification: use `decode.word_mode=char_ctc` or provide larger lexicon.

## Notes
- Baseline wordification is intentionally simple and robust, not SOTA.
- Optional forced alignment (Montreal Forced Aligner): run MFA externally on transcript+audio, then write ARPAbet strings into each internal `.npz` key `phonemes`. Training runs without MFA by using fallback text-to-phoneme heuristics/synthetic labels.

## Consistency Mode (Anti-Collapse Defaults)
- Default config now trains/evals on `utt_*` study samples only.
- Defaults exclude mixed synthetic/sample subsets to reduce output collapse.
- Implemented via `configs/base.yaml -> data.include_prefixes/exclude_prefixes`.
- Trainer warns when batches become blank-dominant (possible CTC collapse).

Recommended clean retrain:
```bash
python scripts/train_phoneme_ctc.py --config configs/train_phoneme_ctc.yaml --max_steps 3000 --save_every_steps 25 --log_every_steps 10 --metrics_file outputs/train_metrics.jsonl
```
