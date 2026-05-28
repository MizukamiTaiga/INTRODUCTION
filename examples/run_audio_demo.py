import os
import sys

# srcモジュールへのパス追加。
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio.spectrogram_sample import generate_spectrogram
from src.audio.gcc_phat_doa_sample import estimate_doa

def main():
    """
    音響処理デモの実行スクリプト。
    """
    print("--- 音響処理デモを開始します ---")
    
    # 用意されたMP3音声のパス設定。
    audio_file = "dummy_oudio.mp3"
    
    if not os.path.exists(audio_file):
        print(f"エラー: {audio_file} が見つかりません。")
        return

    print("\n[1] スペクトログラム生成デモ")
    # スペクトログラム生成の呼び出し。
    generate_spectrogram(audio_file)
    
    print("\n[2] 音源方向推定（DOA）デモ")
    # 音源方向推定の呼び出し。
    estimate_doa()
    
    print("\n--- 音響処理デモが終了しました ---")

if __name__ == "__main__":
    main()
