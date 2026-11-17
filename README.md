# Annotation Tool

BBox와 차선 Segmentation을 편집·승인하고 모델을 추가 학습하는 로컬 웹 도구입니다.

## 설치와 실행

Python 3.10 이상을 사용합니다. GPU를 사용하려면 환경에 맞는 PyTorch/CUDA를 먼저 설치하세요.

```bash
git clone https://github.com/Minjea31/annotation_tool.git
cd annotation_tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run.sh
```

브라우저에서 **http://127.0.0.1:8765**에 접속해 BBox 또는 Segmentation을 선택합니다.
다른 Python이나 포트를 사용하려면 다음처럼 실행합니다.

```bash
ANNOTATION_PYTHON=/path/to/python ANNOTATION_PORT=8766 ./run.sh
```

## 폴더 구조

```text
bbox/
  app.py, engine.py, static/  # 박스 편집·추론·학습
  dataset/                   # BBox 데이터 및 검수 이력 (로컬)
  models/                    # 최신 YOLO26 Nano 가중치와 model.json
  vendor/                    # 동봉한 Ultralytics 8.4.78
segmentation/
  review/                    # 마스크 편집·추론·학습 웹 도구
  train/                     # 학습·평가·내보내기 스크립트
  src/                       # 차선 모델 및 ROS 패키지
  dataset/                   # Segmentation 데이터 및 검수 이력 (로컬)
  models/                    # 최신 MobileNetV3 U-Net 가중치와 model.json
server.py                    # 통합 서버
```

각 `dataset/`은 `unlabeled/`, `train/`, `test/`, `eval/`, `deleted/`, `_store/`로 관리됩니다.
BBox 라벨은 `images/`와 짝을 이루는 `labels/*.txt`(YOLO), Segmentation 라벨은
`images/`와 짝을 이루는 `masks/*.png`입니다. `test/`는 학습 중 검증,
`eval/`은 소속이 고정된 최종 평가에 사용합니다. `_store/`에는 DB·라벨 이력·학습 버전이 저장됩니다.

실제 이미지, 검수 DB, 과거 학습 결과와 캐시는 Git에서 제외합니다.
새로 복제하면 데이터는 비어 있으며 처음 실행할 때 관리 폴더가 생성됩니다.
백업할 때는 각 도구의 `dataset/`과 `models/`를 함께 보관하세요.
데이터 위치는 `BBOX_REVIEW_DATA`, `ANNOTATION_SEG_DATA` 환경변수로 변경할 수 있습니다.

## 사용 방법

1. **＋ 데이터셋 추가**에서 이미지 폴더 경로를 입력합니다. 이미지는 복사하고 중복은 건너뜁니다.
2. 모델을 선택해 추론한 뒤 박스 또는 마스크를 수정합니다. 수동으로도 라벨을 작성할 수 있습니다.
3. **저장** 후 **승인**하면 학습용 데이터에 반영됩니다. 삭제한 이미지는 `deleted/`에 보관됩니다.
4. 저장·승인한 이미지가 최소 5장 모이면 데이터셋 생성 및 추가 학습을 실행할 수 있습니다.
   두 도구의 GPU 작업은 순서대로 실행됩니다.

BBox 주요 단축키: `D` 박스 추가, `A` 선택·이동, `S`/`Delete` 박스 삭제,
`1`~`8` 클래스 변경, `Z`/`X` 실행 취소·다시 실행, `V` 저장, `W` 승인·다음, `Q`/`E` 이전·다음.
휠로 확대하고 Space+드래그로 이동합니다.
클래스 순서는 `slow, bump, crosswalk, left, right, parking, red, green`입니다.

## 포함 모델

- **BBox:** `bbox/models/yolo26n_red_green_200ep.pt` — 가장 최근 등록한 YOLO26 Nano 신호등 모델(red/green, 200 epochs).
  다른 클래스는 라벨 작성 후 추가 학습할 수 있습니다.
- **Segmentation:** `segmentation/models/best.pt` — 가장 최근 완료된 추가 학습의 best 체크포인트(표시된 검증 차선 IoU 약 95.6%).
  마스크 값은 `0` 배경, `1` orange_lane, `2` white_dashed_lane, `255` 무시입니다.

배포 가중치는 도구별 1개이며 출처와 SHA-256은 각 `models/model.json`에 기록했습니다.
새로 복제한 환경의 Segmentation 추가 학습은 **처음부터 학습**으로 시작합니다.
기존 검수 DB가 있는 환경에서는 같은 데이터 정책에 연결된 모델의 이어 학습을 사용할 수 있습니다.

## 검증

```bash
OMP_NUM_THREADS=4 python -m unittest bbox.test_bbox -v
OMP_NUM_THREADS=4 python -m unittest discover -s segmentation/train -p test_training.py -v
```
