import numpy as np
import librosa


def temporal_entropy(y, frame_length=2048, hop_length=512):
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    entropy = []
    for frame in frames.T:
        p = np.abs(frame) / (np.sum(np.abs(frame)) + 1e-10)
        entropy.append(-np.sum(p * np.log2(p + 1e-10)))
    return np.array(entropy)


def spectral_entropy(y, sr, frame_length=2048, hop_length=512):
    stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    entropy = []
    for frame in stft.T:
        p = frame / (np.sum(frame) + 1e-10)
        entropy.append(-np.sum(p * np.log2(p + 1e-10)))
    return np.array(entropy)


def phase_entropy(y, sr, frame_length=2048, hop_length=512):
    stft  = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    phase = np.angle(stft)
    entropy = []
    for frame in phase.T:
        p = np.abs(frame) / (np.sum(np.abs(frame)) + 1e-10)
        entropy.append(-np.sum(p * np.log2(p + 1e-10)))
    return np.array(entropy)


def arr_stats(arr):
    return {
        'mean': float(np.mean(arr)),
        'std':  float(np.std(arr)),
        'q25':  float(np.percentile(arr, 25)),
        'q75':  float(np.percentile(arr, 75)),
        'min':  float(np.min(arr)),
        'max':  float(np.max(arr)),
    }


def extract_entropy_features(audio_path, sr=24000,
                              frame_length=2048, hop_length=512):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    try:
        y, _ = librosa.effects.trim(y, top_db=20)
    except:
        pass
    if y.size == 0:
        return None

    t_ent = temporal_entropy(y, frame_length, hop_length)
    s_ent = spectral_entropy(y, sr, frame_length, hop_length)
    p_ent = phase_entropy(y, sr, frame_length, hop_length)

    feats = {}
    for name, arr in [('temporal', t_ent), ('spectral', s_ent), ('phase', p_ent)]:
        for k, v in arr_stats(arr).items():
            feats[f'{name}_{k}'] = v

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                 n_fft=frame_length, hop_length=hop_length)
    for i in range(13):
        feats[f'mfcc{i+1}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc{i+1}_std']  = float(np.std(mfcc[i]))

    zcr     = librosa.feature.zero_crossing_rate(
                  y, frame_length=frame_length, hop_length=hop_length)[0]
    rms     = librosa.feature.rms(
                  y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent    = librosa.feature.spectral_centroid(
                  y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(
                  y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    for name, arr in [('zcr', zcr), ('rms', rms),
                      ('centroid', cent), ('rolloff', rolloff)]:
        feats[f'{name}_mean'] = float(np.mean(arr))
        feats[f'{name}_std']  = float(np.std(arr))

    feats['path'] = audio_path
    return feats
      
