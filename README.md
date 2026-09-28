# Intrusion Detection System

A file-analysis pipeline that classifies files as **benign** or **malicious**
using a deep neural network built from scratch, with a real-time web
dashboard for non-technical review.

Static-analysis features (byte entropy, printable/null-byte ratios, PE
header structure) are extracted from any file and passed through a custom
feedforward neural network — implemented from first principles in NumPy
(forward pass, backprop, Adam optimizer, gradient clipping, early stopping)
rather than a single call into a black-box library.

## Highlights

- **Feature extraction** (`src/feature_extraction.py`) works on any file
  type. Generic byte-level statistics apply universally; PE (Windows
  executable) files additionally get structural analysis — section counts,
  import-table size, suspicious/packed section names, entry-point
  placement — the same signal categories real static-analysis engines use.
- **Deep neural network** (`src/model.py`) — a from-scratch NumPy
  implementation (no TensorFlow/PyTorch dependency), so every part of the
  training loop is visible and auditable.
- **Dashboard** (`dashboard/`) — Flask app with drag-and-drop file upload,
  live verdict + confidence, per-scan feature breakdown, a running scan
  history, and the model's validation metrics/plots surfaced for
  non-technical stakeholders.
- **Test suite** (`tests/`) covering feature extraction and the model.

## Project structure

```
intrusion-detection-system/
├── data/
│   └── generate_dataset.py    # synthetic training data generator
├── src/
│   ├── feature_extraction.py  # file -> feature vector
│   ├── model.py                # DeepNeuralNetwork (NumPy, from scratch)
│   ├── scaler.py                # StandardScaler
│   ├── train.py                 # training pipeline + evaluation + plots
│   └── predict.py               # CLI: scan a single file
├── dashboard/
│   ├── app.py                   # Flask backend
│   ├── templates/index.html
│   └── static/{style.css, app.js}
├── tests/
│   ├── test_feature_extraction.py
│   └── test_model.py
├── models/                      # trained model + metrics (generated)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Generate the training data** (synthetic — see note below):

```bash
python data/generate_dataset.py
```

**2. Train the model:**

```bash
python src/train.py
```

This trains the network, evaluates it on a held-out test split, and saves
`models/ids_model.npz`, `models/scaler.npz`, `models/metrics.json`, and
`models/training_report.png` (loss curve + confusion matrix).

**3. Scan a file from the command line:**

```bash
python src/predict.py /path/to/some/file.exe
```

**4. Run the dashboard:**

```bash
python dashboard/app.py
```

Then open **http://localhost:5000**, and drag a file onto the drop zone to
scan it in real time.

**5. Run the tests:**

```bash
python -m unittest discover -s tests -v
```

## Results

On a held-out test split of the synthetic dataset (900 samples):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.6%  |
| Precision | 99.1%  |
| Recall    | 96.2%  |
| F1 score  | 97.6%  |

See `models/training_report.png` after training for the loss curve and
confusion matrix.

### Known limitation: entropy alone can't fully separate compressed benign files from packed malware

Compression and encryption both push byte-level entropy toward the same
near-random ceiling (~7.5–8.0). That means an ordinary JPG, PNG, PDF, ZIP,
or MP4 is statistically close to a packed/encrypted malicious payload on
entropy alone — this is a real, well-known limitation of entropy-based
detection, not specific to this project. `data/generate_dataset.py`
models this explicitly: ~35% of the benign training class is high-entropy
"everyday compressed files," so the model doesn't just learn "high
entropy = malicious." For non-PE files this keeps classification honest
but imperfect. For PE (Windows executable) files the model relies much
more heavily on structural features — import-table size, suspicious
section names, high-entropy *sections*, entry-point placement — which
remain strong, reliable signals regardless of overall file entropy (see
the PE-profile sanity checks used during development, in the repo commit
history / README).

## A note on the training data

Real malware corpora (EMBER, VirusShare, etc.) are gigabytes and require
registration/licensing, which doesn't fit a portable, from-scratch repo.
`data/generate_dataset.py` instead samples feature vectors from
distributions that reflect well-documented static-analysis signal (e.g.
packed/encrypted malware shows much higher byte entropy and smaller
resolved-import tables than benign software). The model, training
pipeline, and dashboard are all real and fully functional — swap in
features extracted from a real labeled corpus via
`src/feature_extraction.py` and `train.py` works unchanged.

## License

MIT — see `LICENSE`.
