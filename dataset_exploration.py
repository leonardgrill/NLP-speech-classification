import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf # Alternative to librosa.load for direct access to audio data
from IPython.display import Audio

def dataset_exploration():
    
    audio_file_path = 'speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav'
    # Loading the audio file
    # audio_data, sample_rate = librosa.load(audio_file_path, sr=None) # sr=None to keep the original sample rate
    audio_data, sample_rate = sf.read(audio_file_path) # Alternative for pure data access
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Number of samples: {len(audio_data)}")

    # 6.2 Data Exploration and Preprocessing (Single/Pair) 6 PRACTICAL IMPLEMENTATION
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
    # Visualize waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y=audio_data, sr=sample_rate)
    plt.title('Waveform of the audio file')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    # Play audio file (if in Jupyter Notebook)
    Audio(data=audio_data, rate=sample_rate)