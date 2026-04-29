import joblib
from MFCC_extraction import extract_mfccs
import numpy as np
import os


if __name__ == "__main__":
    #my_audio_path = "own_recordings/left02.wav"  # Adjust the path to your audio file in the own_recordings directory to classify specific audio file

    svm_model = joblib.load('models/svm_model_10-labels.joblib')
    label_encoder_sk = joblib.load('models/label_encoder_10-labels.joblib')

    for audio_file in os.listdir("own_recordings"):
        if audio_file.endswith('.wav'):
            my_audio_path = os.path.join("own_recordings", audio_file)

            my_mfccs = extract_mfccs(my_audio_path, sr=16000, n_mfcc=13)

            my_mfccs_processed_sk = np.mean(my_mfccs, axis=1).reshape(1, -1)

            predicted_label_index_sk = svm_model.predict(my_mfccs_processed_sk)

            predicted_word_sk = label_encoder_sk.inverse_transform(predicted_label_index_sk)

            print(f"File: {audio_file}, Predicted word: {predicted_word_sk}")