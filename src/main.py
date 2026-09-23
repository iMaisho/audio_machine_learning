from pathlib import Path
from src.extract_features import get_feature
import math
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.train_model import train_model

directory = Path("C:/Users/anton/OneDrive/Documents/GitHub/audio_machine_learning/Data/genres_original")

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

features_array_size= len(features)
labels_array_size = len(labels)

assert features_array_size == labels_array_size, "The labels and features are desynchronized. Please try again or contact support if it keeps happening."

print()
print(f"Extraction terminee : {features_array_size} fichiers traites avec succes, {len(failed_files)} en erreur.")
if failed_files:
    print("Fichiers en erreur :")
    for f in failed_files:
        print(f"  - {f}")

permutations = np.random.permutation(features_array_size)
features = np.array(features)[permutations]
labels = np.array(labels)[permutations]

train_index_end = validate_index_start = math.ceil(features_array_size*0.6)
validate_index_end = test_index_start = math.ceil(features_array_size*0.8)

features_train = features[0:train_index_end]
labels_train = labels[0:train_index_end]

features_val = features[validate_index_start:validate_index_end]
labels_val = labels[validate_index_start:validate_index_end]

features_test = features[test_index_start:features_array_size]
labels_test = labels[test_index_start:features_array_size]

train_model(genres, features_train, labels_train, features_val, labels_val)