# :white_check_mark: Road Sign Detection Mini-Project
## 1. 👨‍🏫Project Introduction👨‍🏫
### 1-1. Project Purpose
#### :speaker: TensorRT 사용에 따른 YOLO detection 모델 Inference 결과 비교
##### Data 수: 877 장 이미지
##### 4-classes(0: 'Trafic_light', 1: 'Speedlimit', 2: 'Crosswalk', 3: 'Stop')
---

## 2. ❗Prerequisite & Installation❗
1. 가상환경명 : yolov8
2. Python 버전 : python 3.8

```
pip install ultralytics
pip install opencv-python
pip install matplotlib
```
3.  데이터셋 다운로드
[링크](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)

4. 워크 스페이스 이름: '**project_road_sign_detection**'

## 3. ✅Usage✅
1-1. `road_sign_root`에 데이터셋 담겨있는 폴더 경로 넣어주기
```
road_sign_root = 
annot_path = os.path.join(road_sign_root,"annotations")
img_path = os.path.join(road_sign_root,"images")
label_path = os.path.join(road_sign_root,"labels")
```

1-2. Pascal VOC 포멧에서 YOLO 데이터 포멧으로 변환
```
# Pascal VOC Format to YOLO Format
def xml_to_yolo_bbox(bbox, w, h):
    # Bounding box info
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return (x_center, y_center, width, height)
...

# Normalize BBox info
yolo_bbox = xml_to_yolo_bbox(bbox, width, height)   # [x_center, y_center, width, height]
bbox_string = " ".join([str(x) for x in yolo_bbox])
result.append(f"{index} {bbox_string}")
if result:
    with open(os.path.join(label_path, f"{filename}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(result))
```

1-3. 데이터 split하기
```
test_ratio = 0.1
test_list = file_list[:int(len(file_list)*test_ratio)]
train_list = file_list[int(len(file_list)*test_ratio):]

print(f"train의 개수 : {len(train_list)}, test의 개수 : {len(test_list)}")
# train의 개수 : 790, test의 개수 : 87
```

1-4. Config file(.yaml) 만들기
```
names:
- Trafic_light
- Speedlimit
- Crosswalk
- Stop
nc: 4
test: /content/road_sign_detection/val
train: /content/road_sign_detection/train
val: /content/road_sign_detection/val
```

2. Training & Testing(Validation)
```
from ultralytics import YOLO
# Training
model = YOLO('yolov8s.yaml')
results = model.train(data ='road_sign.yaml', epochs = 100, batch=32, device = 0, patience=30, name='road_sign_s')
...
# Testing
model_path = 
model = YOLO(model_path)  # load a custom model
metrics = model.val()  # no arguments needed, dataset and settings remembered
```

4. TensorRT 변환
* 학습한 모델 가중치 -> **FP32** engine, **FP16** engine
```
# best_fp32.engine 변환
model_path = "/content/drive/MyDrive/Colab_Notebooks/FastCampus/[practice5]road_sign_detection.pt"
model = YOLO(model_path)
model.export(format='engine', device=0, half=False)

# best_fp16.engine 변환
model_path = "/content/drive/MyDrive/Colab_Notebooks/FastCampus/[practice5]road_sign_detection.pt"
model = YOLO(model_path)
model.export(format='engine', device=0, half=True)
```

5. Inference 결과 비교
* TensorRT 적용 전
<div align=center> 
      <img src="https://github.com/user-attachments/assets/82160240-1329-4086-8ad1-24104ca2ecd2" width ="800">
</div>

* FP32 엔진으로 변환
<div align=center> 
      <img src="https://github.com/user-attachments/assets/50bc70ec-890a-414b-b5c1-34756dff75a8" width ="800">
</div>

* FP16 엔진으로 변환
<div align=center> 
      <img src="https://github.com/user-attachments/assets/6b4c768f-e860-4647-86be-69191fa83bfa" width ="800">
</div>
