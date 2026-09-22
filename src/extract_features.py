import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

directory = Path("C:/Users/anton/OneDrive/Documents/GitHub/audio_machine_learning/Data/genres_original")

def get_feature(file_path):
  y, sr = librosa.load(file_path)

  # Extracting MFCC feature
  mfcc = librosa.feature.mfcc(y=y, sr=sr)
  mfcc_mean = mfcc.mean(axis=1)
  mfcc_min = mfcc.min(axis=1)
  mfcc_max = mfcc.max(axis=1)
  mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

  # Extracting Mel Spectrogram feature
  melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
  melspectrogram_mean = melspectrogram.mean(axis=1)
  melspectrogram_min = melspectrogram.min(axis=1)
  melspectrogram_max = melspectrogram.max(axis=1)
  melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

  # Extracting chroma vector feature
  chroma = librosa.feature.chroma_stft(y=y, sr=sr)
  chroma_mean = chroma.mean(axis=1)
  chroma_min = chroma.min(axis=1)
  chroma_max = chroma.max(axis=1)
  chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

  # Extracting tonnetz feature
  tntz = librosa.feature.tonnetz(y=y, sr=sr)
  tntz_mean = tntz.mean(axis=1)
  tntz_min = tntz.min(axis=1)
  tntz_max = tntz.max(axis=1)
  tntz_feature = np.concatenate( (tntz_mean, tntz_min, tntz_max) )

  feature = np.concatenate( (chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
  return feature

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae','rock']
features = []
labels = []
failed_files = []
for genre in genres:
    print("Calculating features for genre : " + genre)
    genre_dir = directory / genre
    for file_path in genre_dir.iterdir():
        try :
            feature = get_feature(file_path)
            features.append(feature)
            label = genres.index(genre)
            labels.append(label)
        except Exception as e:
            print(f'Error in file {file_path}. The file has been skipped')
            failed_files.append(file_path)

print()
print(f"Extraction terminee : {len(features)} fichiers traites avec succes, {len(failed_files)} en erreur.")
if failed_files:
    print("Fichiers en erreur :")
    for f in failed_files:
        print(f"  - {f}")