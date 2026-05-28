import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def generate_spectrogram(audio_path):
    """
    音声データからLinear (STFT) およびLog-Melスペクトログラムを生成するデモ。
    """
    # 音声ファイルの読み込み。
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"音声ファイルの読み込みに失敗しました: {e}")
        return

    # Linear (STFT) スペクトログラムの生成。
    D = librosa.stft(y, n_fft=1024)
    # 振幅をデシベルスケールに変換。
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Log-Melスペクトログラムの生成。
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, n_mels=64)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # スペクトログラムの画像としての保存。
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear Spectrogramの描画。
    img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', sr=sr, ax=axes[0])
    axes[0].set_title('Linear (STFT) Spectrogram')
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")
    
    # Log-Mel Spectrogramの描画。
    img2 = librosa.display.specshow(log_mel_spec, x_axis='time', y_axis='mel', sr=sr, ax=axes[1])
    axes[1].set_title('Log-Mel Spectrogram')
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    
    import os
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = f"{base_name}_spectrogram.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"スペクトログラムを {output_path} に保存しました。")

if __name__ == "__main__":
    # デモ用の音声ファイルパス。
    test_audio = "dummy_oudio.mp3"
    
    # 処理の実行。
    generate_spectrogram(test_audio)
