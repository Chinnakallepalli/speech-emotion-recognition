import librosa
import numpy as np

def extract_feature_logic(X, sample_rate):
    """Extracted logic to convert raw audio into the 182 features the model expects."""
    X = librosa.util.normalize(X)
    result = np.array([])
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    result = np.hstack((result, np.mean(mfccs, axis=1), np.std(mfccs, axis=1)))
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
    result = np.hstack((result, np.mean(chroma, axis=1)))
    
    # Mel
    mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    result = np.hstack((result, np.mean(mel_db, axis=1)))
    
    # Energy & Temporal (ZCR, RMS, Rolloff)
    zcr = librosa.feature.zero_crossing_rate(y=X)
    rms = librosa.feature.rms(y=X)
    rolloff = librosa.feature.spectral_rolloff(y=X, sr=sample_rate)
    result = np.hstack((result, np.mean(zcr), np.std(zcr), np.mean(rms), np.mean(rolloff)))
    
    return result