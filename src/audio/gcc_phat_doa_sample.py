import numpy as np

def calculate_gcc_phat(sig1, sig2, fs=16000):
    """
    GCC-PHATを用いた相互相関の計算および遅延の推定。
    """
    # フーリエ変換の実行。
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)

    # クロススペクトルの計算。
    R = SIG1 * np.conj(SIG2)

    # PHAT重み付けの適用。
    cc = np.fft.irfft(R / np.abs(R), n=n)

    # 最大相関となるシフト量の取得。
    max_shift = int(n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(cc) - max_shift

    # 到達時間差の算出。
    delay = shift / float(fs)
    return delay

def estimate_doa():
    """
    推定遅延からの音源方向推定デモ。
    """
    # 疑似的な多チャンネル信号の生成。
    fs = 16000
    t = np.linspace(0, 1, fs)
    source_signal = np.sin(2 * np.pi * 440 * t)
    
    # 意図的な遅延の付与（シミュレーション）。
    # ※遅延サンプル数とは、音がマイク1に到達してからマイク2に到達するまでの時間差を、
    # サンプリング周波数（1秒あたりのデータ数）に基づく「データ数のズレ」として表したものです。
    # 例えば16000Hzで15サンプルのズレは、15 / 16000 = 約0.0009秒の遅延を意味します。
    delay_samples = 15
    sig1 = source_signal
    sig2 = np.roll(source_signal, delay_samples)

    # GCC-PHATによる遅延推定。
    estimated_delay = calculate_gcc_phat(sig1, sig2, fs)
    estimated_samples = int(estimated_delay * fs)

    # 推定結果の表示。
    print(f"真の遅延サンプル数: {delay_samples}")
    print(f"推定された遅延サンプル数: {estimated_samples}")
    
    # 簡略化した角度の算出（本来はマイク間隔と音速から計算）。
    angle = np.degrees(np.arcsin(min(max(estimated_delay * 343 / 0.1, -1), 1)))
    print(f"推定音源方向（角度）: 約 {angle:.2f} 度")

if __name__ == "__main__":
    # 音源方向推定の実行。
    estimate_doa()
