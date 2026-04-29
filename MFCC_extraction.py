import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf # Alternative to librosa.load for direct access to audio data


def extract_mfccs(audio_path, sr=16000, n_mfcc=256, n_fft=4096, hop_length=512):
    """Extracts MFCCs from an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate (default: 16000 Hz for speech commands).
        n_mfcc (int): Number of MFCCs to extract.
        n_fft (int): Size of the FFT window.
        hop_length (int): Step size between consecutive frames.

    Returns:
        numpy.ndarray: Array of MFCCs (shape: n_mfcc, number_of_frames).
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfccs

if __name__ == "__main__":
    audio_file_path = 'speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav'
    audio_data, sample_rate = sf.read(audio_file_path)
    # Example of MFCC extraction and visualization
    mfccs = extract_mfccs(audio_file_path)
    print(f"Shape of the MFCCs: {mfccs.shape}")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, hop_length=512)
    plt.colorbar()
    plt.title('MFCCs')
    plt.tight_layout()
    plt.show()