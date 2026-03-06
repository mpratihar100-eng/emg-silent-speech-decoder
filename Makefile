PYTHON=python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

download:
	$(PYTHON) scripts/download_data.py --config configs/base.yaml --source zenodo_vss

train-smoke:
	$(PYTHON) scripts/train_phoneme_ctc.py --config configs/train_phoneme_ctc.yaml --max_steps 20

eval:
	$(PYTHON) scripts/eval_phoneme_ctc.py --config configs/eval.yaml

infer:
	$(PYTHON) scripts/infer_phoneme.py --config configs/infer.yaml --split test --num_samples 3

stream-sim:
	$(PYTHON) scripts/infer_stream.py --config configs/hardware_serial.yaml --simulate

visualize:
	$(PYTHON) scripts/visualize_emg.py --config configs/infer.yaml --split test --index 0

sample-template:
	$(PYTHON) scripts/prepare_sample_data.py --config configs/sample_import.yaml --create_template

sample-import:
	$(PYTHON) scripts/prepare_sample_data.py --config configs/sample_import.yaml

test:
	pytest -q
