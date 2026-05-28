import cv2
from ultralytics import YOLO

def run_detection(image_path, model_path="yolo11n.pt"):
    """
    YOLOを用いたランドマーク検出のデモ。
    """
    # YOLOモデルの読み込み。
    model = YOLO(model_path)

    # 入力画像の読み込み。
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像が見つかりませんでした: {image_path}")
        return

    # YOLOによる物体検出の実行。
    results = model(img)

    # 検出結果の解析と描画。
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # バウンディングボックスの座標取得。
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # クラスIDと信頼度の取得。
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # バウンディングボックスの描画。
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # クラス名と信頼度のテキスト描画。
            label = f"{model.names[cls_id]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 検出結果の出力。
    print("検出が完了しました。")
    
    import os
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_detection_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"結果を {output_path} に保存しました。")

if __name__ == "__main__":
    # サンプル画像のパス指定。
    sample_img = "dummy_image.jpg"
    
    # 画像が存在しない場合は、サンプルの実画像をダウンロード。
    import os
    if not os.path.exists(sample_img):
        import urllib.request
        print("テスト用のサンプル画像をダウンロードしています...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", sample_img)
    
    # 処理の実行。
    run_detection(sample_img)
