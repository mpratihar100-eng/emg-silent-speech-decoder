from __future__ import annotations

import csv
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    HAS_DND = True
except Exception:
    DND_FILES = "DND_Files"  # type: ignore
    TkinterDnD = None  # type: ignore
    HAS_DND = False


ROOT = Path(__file__).resolve().parent
SAMPLE_RAW = ROOT / "data" / "sample_raw"


def parse_drop_payload(payload: str) -> list[Path]:
    raw = payload.strip()
    if not raw:
        return []
    try:
        parts = shlex.split(raw, posix=False)
    except Exception:
        parts = raw.split()
    out: list[Path] = []
    for p in parts:
        p = p.strip().strip("{}").strip('"')
        if p:
            out.append(Path(p))
    return out


class BaseWindow(TkinterDnD.Tk if HAS_DND else tk.Tk):  # type: ignore[misc]
    pass


class App(BaseWindow):
    def __init__(self) -> None:
        super().__init__()
        self.title("EMG Silent Speech Decoder - Tabbed App")
        self.geometry("1080x760")
        self.minsize(980, 680)

        self.python_bin = tk.StringVar(value=sys.executable)
        self.cfg_base = tk.StringVar(value="configs/base.yaml")
        self.cfg_train = tk.StringVar(value="configs/train_phoneme_ctc.yaml")
        self.cfg_eval = tk.StringVar(value="configs/eval.yaml")
        self.cfg_infer = tk.StringVar(value="configs/infer.yaml")
        self.cfg_hw = tk.StringVar(value="configs/hardware_serial.yaml")
        self.cfg_sample = tk.StringVar(value="configs/sample_import.yaml")

        self.max_steps = tk.StringVar(value="20")
        self.num_samples = tk.StringVar(value="5")
        self.decode_mode = tk.StringVar(value="beam")
        self.stream_windows = tk.StringVar(value="10")
        self.stream_file = tk.StringVar(value="")
        self.viz_file = tk.StringVar(value="")
        self.viz_split = tk.StringVar(value="test")
        self.viz_index = tk.StringVar(value="0")

        self.last_process: subprocess.Popen[str] | None = None
        self.dropped_files: list[Path] = []

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Python:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.python_bin, width=80).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Browse", command=self._choose_python).pack(side=tk.LEFT)
        ttk.Button(top, text="Stop Running", command=self.stop_current).pack(side=tk.LEFT, padx=8)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.tab_start = ttk.Frame(self.nb, padding=10)
        self.tab_data = ttk.Frame(self.nb, padding=10)
        self.tab_train = ttk.Frame(self.nb, padding=10)
        self.tab_infer = ttk.Frame(self.nb, padding=10)
        self.tab_visualize = ttk.Frame(self.nb, padding=10)
        self.tab_stream = ttk.Frame(self.nb, padding=10)
        self.tab_config = ttk.Frame(self.nb, padding=10)
        self.tab_logs = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_start, text="Start")
        self.nb.add(self.tab_data, text="Data")
        self.nb.add(self.tab_train, text="Train/Eval")
        self.nb.add(self.tab_infer, text="Infer")
        self.nb.add(self.tab_visualize, text="Visualize")
        self.nb.add(self.tab_stream, text="Stream")
        self.nb.add(self.tab_config, text="Settings")
        self.nb.add(self.tab_logs, text="Logs")

        self._build_start_tab()
        self._build_data_tab()
        self._build_train_tab()
        self._build_infer_tab()
        self._build_visualize_tab()
        self._build_stream_tab()
        self._build_config_tab()
        self._build_logs_tab()

    def _build_start_tab(self) -> None:
        ttk.Label(self.tab_start, text="One app, tabbed workflow.", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(
            self.tab_start,
            text=(
                "Recommended order: Settings -> Data -> Train/Eval -> Infer/Visualize/Stream.\n"
                "Use Data tab to drag/drop your files and import them."
            ),
            wraplength=980,
        ).pack(anchor=tk.W, pady=(6, 12))

        row = ttk.Frame(self.tab_start)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Install Requirements", command=self.run_setup).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Download Baseline Data", command=self.run_download).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Preprocess + Align", command=self.run_preprocess).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Open Logs Tab", command=lambda: self.nb.select(self.tab_logs)).pack(side=tk.LEFT, padx=4)

        proc = ttk.LabelFrame(self.tab_start, text="Processing Pipeline", padding=10)
        proc.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(
            proc,
            text=(
                "Input file -> parse [T,C] -> internal .npz -> preprocess (notch/bandpass/normalize) -> "
                "DTW template alignment manifest -> feature extraction (raw or STFT) -> "
                "CTC phoneme decoding -> lexicon word mapping"
            ),
            wraplength=980,
        ).pack(anchor=tk.W)

    def _build_data_tab(self) -> None:
        top = ttk.Frame(self.tab_data)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Create Sample Manifest", command=self.run_sample_template).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Import Dropped Files", command=self.run_import_dropped).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Import Existing sample_raw", command=self.run_sample_import).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Preprocess Corpus", command=self.run_preprocess).pack(side=tk.LEFT, padx=4)

        drop_frame = ttk.LabelFrame(self.tab_data, text="Drag & Drop Sample Data", padding=10)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        ttk.Label(
            drop_frame,
            text="Drop .csv/.npy/.npz or folders/archives (.zip/.tar.gz). Then click 'Import Dropped Files'.",
            wraplength=980,
        ).pack(anchor=tk.W)

        self.drop_target = tk.Text(drop_frame, height=8, wrap=tk.WORD)
        self.drop_target.pack(fill=tk.BOTH, expand=True, pady=8)
        self.drop_target.insert(tk.END, "Drop here or use Add Files/Add Folder.")
        self.drop_target.configure(state=tk.DISABLED)

        row = ttk.Frame(drop_frame)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Add Files", command=self.pick_files).pack(side=tk.LEFT)
        ttk.Button(row, text="Add Folder", command=self.pick_folder).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Clear List", command=self.clear_dropped).pack(side=tk.LEFT)

        if HAS_DND:
            self.drop_target.drop_target_register(DND_FILES)
            self.drop_target.dnd_bind("<<Drop>>", self.on_drop)
        else:
            ttk.Label(drop_frame, text="Drag-drop disabled: install tkinterdnd2 to enable.").pack(anchor=tk.W, pady=(8, 0))

    def _build_train_tab(self) -> None:
        row = ttk.Frame(self.tab_train)
        row.pack(fill=tk.X)
        ttk.Label(row, text="max_steps").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.max_steps, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Train Quick", command=self.run_train).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Evaluate", command=self.run_eval).pack(side=tk.LEFT, padx=4)

    def _build_infer_tab(self) -> None:
        row = ttk.Frame(self.tab_infer)
        row.pack(fill=tk.X)
        ttk.Label(row, text="num_samples").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.num_samples, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="decode").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.decode_mode, values=["beam", "greedy"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Run Inference", command=self.run_infer).pack(side=tk.LEFT, padx=4)

    def _build_visualize_tab(self) -> None:
        row1 = ttk.Frame(self.tab_visualize)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="split").pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.viz_split, values=["train", "val", "test"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Label(row1, text="index").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.viz_index, width=8).pack(side=tk.LEFT, padx=6)

        row2 = ttk.Frame(self.tab_visualize)
        row2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row2, text="npz file (optional)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.viz_file, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(row2, text="Browse", command=self._choose_viz_file).pack(side=tk.LEFT)

        row3 = ttk.Frame(self.tab_visualize)
        row3.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(row3, text="Generate Visualizations", command=self.run_visualize).pack(side=tk.LEFT)
        ttk.Label(row3, text="Outputs: outputs/visuals/*_emg.png and *_phoneme_timeline.png").pack(side=tk.LEFT, padx=8)

    def _build_stream_tab(self) -> None:
        row1 = ttk.Frame(self.tab_stream)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="max_windows").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.stream_windows, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Run Stream Sim", command=self.run_stream).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(self.tab_stream)
        row2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row2, text="simulate_file (optional)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.stream_file, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(row2, text="Browse", command=self._choose_stream_file).pack(side=tk.LEFT)

    def _build_config_tab(self) -> None:
        rows = [
            ("Base config", self.cfg_base),
            ("Train config", self.cfg_train),
            ("Eval config", self.cfg_eval),
            ("Infer config", self.cfg_infer),
            ("Hardware config", self.cfg_hw),
            ("Sample import config", self.cfg_sample),
        ]
        for label, var in rows:
            r = ttk.Frame(self.tab_config)
            r.pack(fill=tk.X, pady=3)
            ttk.Label(r, text=label, width=20).pack(side=tk.LEFT)
            ttk.Entry(r, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            self.tab_config,
            text="You can point these to custom YAML files. Buttons in other tabs use these paths.",
        ).pack(anchor=tk.W, pady=(10, 0))

    def _build_logs_tab(self) -> None:
        self.log = tk.Text(self.tab_logs, wrap=tk.NONE)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state=tk.DISABLED)

        row = ttk.Frame(self.tab_logs)
        row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(row, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT)

    def _choose_python(self) -> None:
        path = filedialog.askopenfilename(title="Choose python executable")
        if path:
            self.python_bin.set(path)

    def _choose_stream_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose stream .npz file", filetypes=[("NPZ", "*.npz"), ("All", "*.*")])
        if path:
            self.stream_file.set(path)

    def _choose_viz_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose internal .npz sample", filetypes=[("NPZ", "*.npz"), ("All", "*.*")])
        if path:
            self.viz_file.set(path)

    def _set_drop_text(self, text: str) -> None:
        self.drop_target.configure(state=tk.NORMAL)
        self.drop_target.delete("1.0", tk.END)
        self.drop_target.insert(tk.END, text)
        self.drop_target.configure(state=tk.DISABLED)

    def _refresh_drop_preview(self) -> None:
        if not self.dropped_files:
            self._set_drop_text("Drop here or use Add Files/Add Folder.")
            return
        lines = [str(p) for p in self.dropped_files[:20]]
        extra = len(self.dropped_files) - len(lines)
        if extra > 0:
            lines.append(f"... and {extra} more")
        self._set_drop_text("\n".join(lines))

    def pick_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Choose sample files",
            filetypes=[("Data", "*.csv *.npy *.npz *.zip *.tar *.gz"), ("All", "*.*")],
        )
        if files:
            self.dropped_files.extend(Path(f) for f in files)
            self._refresh_drop_preview()

    def pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose folder containing sample files")
        if folder:
            self.dropped_files.append(Path(folder))
            self._refresh_drop_preview()

    def clear_dropped(self) -> None:
        self.dropped_files = []
        self._refresh_drop_preview()

    def on_drop(self, event) -> None:  # type: ignore[no-untyped-def]
        files = parse_drop_payload(event.data)
        if files:
            self.dropped_files.extend(files)
            self._refresh_drop_preview()

    def append_log(self, msg: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def clear_logs(self) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def stop_current(self) -> None:
        if self.last_process and self.last_process.poll() is None:
            self.last_process.terminate()
            self.append_log("\n[INFO] Stopped running command.\n")

    def _run(self, args: list[str]) -> None:
        if self.last_process and self.last_process.poll() is None:
            messagebox.showwarning("Busy", "A command is already running.")
            return
        py = self.python_bin.get().strip()
        if not py:
            messagebox.showerror("Missing Python", "Set python executable path first.")
            return
        cmd = [py] + args
        self.nb.select(self.tab_logs)
        self.append_log(f"\n$ {' '.join(cmd)}\n")

        def worker() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.last_process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.after(0, self.append_log, line)
                code = proc.wait()
                self.after(0, self.append_log, f"\n[EXIT] code={code}\n")
            except Exception as exc:
                self.after(0, self.append_log, f"\n[ERROR] {exc}\n")
            finally:
                self.last_process = None

        threading.Thread(target=worker, daemon=True).start()

    def _collect_supported_files(self, p: Path) -> list[Path]:
        files: list[Path] = []
        if p.is_dir():
            for ext in ("*.csv", "*.npy", "*.npz"):
                files.extend(sorted(p.rglob(ext)))
        elif p.is_file():
            name = p.name.lower()
            if name.endswith((".csv", ".npy", ".npz")):
                files.append(p)
            elif name.endswith(".zip"):
                out = SAMPLE_RAW / "_dropped_extract"
                out.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(out)
                for ext in ("*.csv", "*.npy", "*.npz"):
                    files.extend(sorted(out.rglob(ext)))
            elif name.endswith((".tar", ".tar.gz", ".tgz", ".gz")):
                out = SAMPLE_RAW / "_dropped_extract"
                out.mkdir(parents=True, exist_ok=True)
                try:
                    with tarfile.open(p, "r:*") as tf:
                        tf.extractall(out)
                    for ext in ("*.csv", "*.npy", "*.npz"):
                        files.extend(sorted(out.rglob(ext)))
                except Exception:
                    pass
        return files

    def _stage_dropped_files(self) -> int:
        SAMPLE_RAW.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        for src in self.dropped_files:
            staged.extend(self._collect_supported_files(src))
        if not staged:
            return 0

        rows = []
        for i, src in enumerate(staged):
            dst = SAMPLE_RAW / f"drop_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            rows.append(
                {
                    "file": dst.name,
                    "sr": "1000",
                    "text": "",
                    "phonemes": "",
                    "speaker_id": "drag_user",
                    "session_id": "drag_session",
                    "split": "test",
                }
            )

        manifest = SAMPLE_RAW / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["file", "sr", "text", "phonemes", "speaker_id", "session_id", "split"])
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def run_setup(self) -> None:
        self._run(["-m", "pip", "install", "-r", "requirements.txt"])

    def run_download(self) -> None:
        self._run(["scripts/download_data.py", "--config", self.cfg_base.get(), "--source", "zenodo_vss"])

    def run_preprocess(self) -> None:
        self._run(["scripts/preprocess_align.py", "--config", self.cfg_base.get()])

    def run_sample_template(self) -> None:
        self._run(["scripts/prepare_sample_data.py", "--config", self.cfg_sample.get(), "--create_template"])

    def run_sample_import(self) -> None:
        self._run(["scripts/prepare_sample_data.py", "--config", self.cfg_sample.get()])

    def run_import_dropped(self) -> None:
        n = self._stage_dropped_files()
        if n == 0:
            messagebox.showwarning("No files", "No supported files found. Drop files/folder/archive first.")
            return
        self.append_log(f"\n[INFO] Staged {n} files into data/sample_raw\n")
        self._run(["scripts/prepare_sample_data.py", "--config", self.cfg_sample.get()])

    def run_train(self) -> None:
        steps = self.max_steps.get().strip() or "20"
        self._run(["scripts/train_phoneme_ctc.py", "--config", self.cfg_train.get(), "--max_steps", steps])

    def run_eval(self) -> None:
        self._run(["scripts/eval_phoneme_ctc.py", "--config", self.cfg_eval.get(), "--decode", self.decode_mode.get()])

    def run_infer(self) -> None:
        n = self.num_samples.get().strip() or "5"
        self._run(
            [
                "scripts/infer_phoneme.py",
                "--config",
                self.cfg_infer.get(),
                "--split",
                "test",
                "--num_samples",
                n,
                "--decode",
                self.decode_mode.get(),
            ]
        )

    def run_stream(self) -> None:
        n = self.stream_windows.get().strip() or "10"
        cmd = [
            "scripts/infer_stream.py",
            "--config",
            self.cfg_hw.get(),
            "--simulate",
            "--max_windows",
            n,
            "--decode",
            self.decode_mode.get(),
        ]
        sf = self.stream_file.get().strip()
        if sf:
            cmd.extend(["--simulate_file", sf])
        self._run(cmd)

    def run_visualize(self) -> None:
        cmd = [
            "scripts/visualize_emg.py",
            "--config",
            self.cfg_infer.get(),
            "--split",
            self.viz_split.get(),
            "--index",
            self.viz_index.get().strip() or "0",
        ]
        vf = self.viz_file.get().strip()
        if vf:
            cmd.extend(["--npz_path", vf])
        self._run(cmd)


if __name__ == "__main__":
    app = App()
    app.mainloop()
