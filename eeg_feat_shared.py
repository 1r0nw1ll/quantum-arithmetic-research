"""Shared utilities for EEG feature suite — loaded by all five analysis scripts."""
import re, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pyedflib

warnings.filterwarnings("ignore")

CHBMIT_ROOT = Path(__file__).parent / "archive/phase_artifacts/phase2_data/eeg/chbmit"
WINDOW_SEC  = 10.0
BUFFER_SEC  = 300.0
SEED        = 42
TEST_FRAC   = 0.30
N_CHANNELS  = 23
FS          = 256

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8, 13),
    "beta":  (13,30),
    "gamma": (30,50),
}

def read_edf(path):
    with pyedflib.EdfReader(str(path)) as f:
        n_use = min(f.signals_in_file, N_CHANNELS)
        fs    = int(round(f.getSampleFrequency(0)))
        n_s   = f.getNSamples()[0]
        sig   = np.zeros((n_use, n_s), dtype=np.float32)
        for i in range(n_use): sig[i] = f.readSignal(i)
    if n_use < N_CHANNELS:
        sig = np.pad(sig, ((0, N_CHANNELS - n_use), (0, 0)))
    return sig, fs

def parse_summary(path):
    ann, cur = defaultdict(list), None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m: cur = m.group(1).lower(); continue
            if cur is None: continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second", line, re.I)
            if m: ann[cur].append({"start_s": int(m.group(1)), "end_s": None}); continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                for a in reversed(ann.get(cur, [])):
                    if a["end_s"] is None: a["end_s"] = int(m.group(1)); break
    return {k: [a for a in v if a["end_s"] is not None] for k, v in ann.items() if v}

def extract_windows(patient_dir, max_sei=None, max_bas=None):
    summaries = list(patient_dir.glob("*-summary.txt"))
    if not summaries: return []
    ann = parse_summary(summaries[0])
    if not ann: return []
    sei_wins, bas_wins = [], []
    all_ivs = defaultdict(list)
    for fname, seqs in ann.items():
        for s in seqs: all_ivs[fname].append((float(s["start_s"]), float(s["end_s"])))
    for fname, seqs in ann.items():
        edf = patient_dir / fname
        if not edf.exists(): continue
        try: sig, fs = read_edf(edf)
        except: continue
        total_s = sig.shape[1] / fs
        for s in seqs:
            onset, offset = float(s["start_s"]), float(s["end_s"])
            for i in range(max(1, int((offset - onset) // WINDOW_SEC))):
                t0 = onset + i * WINDOW_SEC
                if t0 + WINDOW_SEC > total_s: break
                s0, s1 = int(t0*fs), int((t0+WINDOW_SEC)*fs)
                sei_wins.append({"label":1,"sig":sig[:,s0:s1],"fs":fs})
        ivs = all_ivs[fname]
        earliest, latest = min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)
        added = 0
        for pos in np.arange(WINDOW_SEC, earliest - BUFFER_SEC, WINDOW_SEC*3):
            if added >= 5: break
            s0,s1 = int(pos*fs), int((pos+WINDOW_SEC)*fs)
            bas_wins.append({"label":0,"sig":sig[:,s0:s1],"fs":fs}); added+=1
        for pos in np.arange(latest + BUFFER_SEC, total_s - WINDOW_SEC, WINDOW_SEC*3):
            if added >= 10: break
            s0,s1 = int(pos*fs), int((pos+WINDOW_SEC)*fs)
            bas_wins.append({"label":0,"sig":sig[:,s0:s1],"fs":fs}); added+=1
    if not sei_wins or not bas_wins: return []
    rng = np.random.default_rng(SEED)
    n = min(len(sei_wins), len(bas_wins))
    return ([sei_wins[i] for i in rng.choice(len(sei_wins), n, replace=False)] +
            [bas_wins[i] for i in rng.choice(len(bas_wins), n, replace=False)])

def bandpass(sig, fs, low, high):
    nyq = fs / 2.0
    lo, hi = max(low/nyq, 1e-3), min(high/nyq, 0.999)
    if lo >= hi: return sig
    b, a = butter(4, [lo, hi], btype='band')
    return filtfilt(b, a, sig.astype(np.float64), axis=1)

def band_power(sig, fs, low, high):
    nperseg = min(2*fs, sig.shape[1])
    rows = []
    for ch in range(sig.shape[0]):
        freqs, Pxx = welch(sig[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs>=low)&(freqs<=high)
        rows.append(float(np.mean(Pxx[mask])) if mask.any() else 0.0)
    return np.array(rows)

def delta_power_scalar(sig, fs):
    return float(band_power(sig, fs, 1, 4).mean())

def _null_ll(y):
    p = np.clip(y.mean(), 1e-10, 1-1e-10)
    return float(y.sum()*np.log(p) + (1-y).sum()*np.log(1-p))

def _model_ll(lr, X, y):
    probs = np.clip(lr.predict_proba(X)[:,1], 1e-10, 1-1e-10)
    return float(np.sum(y*np.log(probs) + (1-y)*np.log(1-probs)))

def eval_delta_r2(feats, labels):
    """Returns (delta_r2, p_lr) for feats vs delta-only baseline."""
    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC,
                                      stratify=labels, random_state=SEED)
    y_tr, y_te = labels[tr_idx], labels[te_idx]
    ll_null = _null_ll(y_te)
    lr_base = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_base.fit(feats[tr_idx,:1], y_tr)
    ll_base = _model_ll(lr_base, feats[te_idx,:1], y_te)
    r2_base = 1.0 - ll_base/ll_null if ll_null else 0.0
    lr_aug  = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_aug.fit(feats[tr_idx], y_tr)
    ll_aug  = _model_ll(lr_aug, feats[te_idx], y_te)
    r2_aug  = 1.0 - ll_aug/ll_null if ll_null else 0.0
    n_qa = feats.shape[1] - 1
    lrs  = max(0.0, 2.0*(ll_aug - ll_base))
    p    = float(chi2.sf(lrs, df=n_qa))
    return float(r2_aug - r2_base), float(p)

def fishers(ps):
    ps = [max(1e-15,p) for p in ps]
    stat = -2.0*sum(np.log(p) for p in ps)
    return float(stat), float(chi2.sf(stat, 2*len(ps)))

def ready_patients():
    return [pd for pd in sorted(CHBMIT_ROOT.glob("chb*/"))
            if list(pd.glob("*-summary.txt")) and not list(pd.glob("*.tmp"))]

def orbit_period_m24(b0, e0, m=24):
    b, e = b0, e0
    for k in range(1, 300):
        b, e = e, (b+e-1)%m+1
        if b==b0 and e==e0: return k
    return -1

def orbit_fam(b, e, m=24):
    p = orbit_period_m24(b, e, m)
    return "Singularity" if p==1 else "Satellite" if p==8 else "Cosmos"
