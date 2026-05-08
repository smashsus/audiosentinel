import os
import numpy as np
import joblib
from .features import extract_entropy_features

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

FEATURE_COLS = [
    'temporal_mean','temporal_std','temporal_q25','temporal_q75','temporal_min','temporal_max',
    'spectral_mean','spectral_std','spectral_q25','spectral_q75','spectral_min','spectral_max',
    'phase_mean','phase_std','phase_q25','phase_q75','phase_min','phase_max',
    'mfcc1_mean','mfcc1_std','mfcc2_mean','mfcc2_std','mfcc3_mean','mfcc3_std',
    'mfcc4_mean','mfcc4_std','mfcc5_mean','mfcc5_std','mfcc6_mean','mfcc6_std',
    'mfcc7_mean','mfcc7_std','mfcc8_mean','mfcc8_std','mfcc9_mean','mfcc9_std',
    'mfcc10_mean','mfcc10_std','mfcc11_mean','mfcc11_std','mfcc12_mean','mfcc12_std',
    'mfcc13_mean','mfcc13_std',
    'zcr_mean','zcr_std','rms_mean','rms_std',
    'centroid_mean','centroid_std','rolloff_mean','rolloff_std',
]

_rf     = None
_scaler = None


def _load_models():
    global _rf, _scaler
    if _rf is None:
        _rf     = joblib.load(os.path.join(_MODEL_DIR, 'rf_best.pkl'))
        _scaler = joblib.load(os.path.join(_MODEL_DIR, 'scaler.pkl'))


def predict_audio(path, verbose=True):
    """
    Predict whether an audio file is Human or AI-generated.

    Parameters
    ----------
    path : str
        Path to a WAV file.
    verbose : bool
        Print result to stdout.

    Returns
    -------
    dict with keys: label (str), pred (int), prob_ai (float), prob_human (float)
    """
    _load_models()

    feats = extract_entropy_features(path)
    if feats is None:
        raise ValueError(f"Could not extract features from: {path}")

    row = np.array([feats.get(col, 0.0) for col in FEATURE_COLS]).reshape(1, -1)
    X   = _scaler.transform(row)

    pred  = int(_rf.predict(X)[0])
    prob  = _rf.predict_proba(X)[0]
    label = "HUMAN" if pred == 1 else "AI"

    if verbose:
        print(f"File       : {os.path.basename(path)}")
        print(f"Result     : {label}")
        print(f"Confidence : {max(prob)*100:.1f}%")
        print(f"P(AI)={prob[0]:.3f}  P(Human)={prob[1]:.3f}")

    return {
        'label':      label,
        'pred':       pred,
        'prob_ai':    float(prob[0]),
        'prob_human': float(prob[1]),
    }


def predict_int(path):
    """
    Returns integer label only: 0 = AI, 1 = Human.

    Parameters
    ----------
    path : str
        Path to a WAV file.

    Returns
    -------
    int
        0 if AI-generated, 1 if Human.
    """
    return predict_audio(path, verbose=False)['pred']


def predict_batch(paths, verbose=False):
    """
    Run predict_audio on a list of WAV paths.

    Returns list of result dicts (None for failed files).
    """
    results = []
    for path in paths:
        try:
            results.append(predict_audio(path, verbose=verbose))
        except Exception as e:
            print(f"Failed: {path} — {e}")
            results.append(None)
    return results
      
