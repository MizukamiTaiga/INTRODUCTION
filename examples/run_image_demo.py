import os
import sys

# srcモジュールへのパス追加。
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.image.landmark_detection_sample import run_detection

def main():
    """
    画像認識デモの実行スクリプト。
    """
    print("--- 画像認識デモを開始します ---")
    
    # ダミー画像のパス設定。
    sample_img = "dummy_image.jpg"
    
    # サンプル画像が存在しない場合はダウンロード。
    if not os.path.exists(sample_img):
        import urllib.request
        print(f"テスト用の画像 {sample_img} をダウンロードしています...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", sample_img)

    # ランドマーク検出の呼び出し。
    run_detection(sample_img)
    print("--- 画像認識デモが終了しました ---")

if __name__ == "__main__":
    main()
