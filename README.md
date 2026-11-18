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
`./run.sh`는 저장소의 `.venv`, 현재 Python, 기존 Conda `yolo` 환경 순서로 필요한 패키지가 있는 Python을 찾습니다.
`launcher.html`을 직접 열면 기본 서버 주소로 이동하므로 서버를 먼저 실행하세요.
다른 Python이나 포트를 사용하려면 다음처럼 실행합니다.

```bash
ANNOTATION_PYTHON=/path/to/python ANNOTATION_PORT=8766 ./run.sh
```

### 같은 Wi-Fi에서 함께 사용하기

`./run.sh`는 기본적으로 `0.0.0.0`에서 접속을 받습니다. 실행한 터미널에 표시되는
**같은 Wi-Fi에서 접속** 주소(예: `http://192.168.0.10:8765`)를 다른 사람에게 공유하세요.
같은 Wi-Fi에 연결된 컴퓨터나 휴대폰의 브라우저에서 해당 주소로 접속하면 함께 작업할 수 있습니다.
서버 컴퓨터에서는 기존 `http://127.0.0.1:8765` 주소도 계속 사용할 수 있습니다.
서버가 이미 실행 중이라면 종료한 뒤 `./run.sh`로 다시 실행해야 변경된 접속 설정이 적용됩니다.

모든 사람의 저장·승인·삭제·학습은 서버를 실행한 컴퓨터에서 기존 로직으로 처리됩니다.
이미지·라벨·검수 DB·학습 결과도 이 컴퓨터의 기존 `bbox/dataset/`, `segmentation/dataset/`에 저장되며,
저장 위치 환경변수를 지정했다면 해당 경로를 사용합니다. 데이터셋 추가에 입력하는 폴더 경로 역시
서버 컴퓨터의 경로입니다. 접속한 사람의 기기에는 서버 데이터가 따로 저장되지 않습니다.
서버를 켜 둔 동안 접속할 수 있습니다.

서로 다른 이미지를 나눠 작업하는 것을 권장합니다. 같은 이미지를 동시에 수정하면
먼저 저장한 결과를 보호하기 위해 뒤에 저장하는 요청은 변경 충돌로 거절됩니다.
Segmentation은 편집 중이 아닐 때 다른 사람이 저장한 최신 라벨을 자동으로 불러옵니다.
편집 중 충돌이 나면 내 수정은 유지되며 **최신 라벨 불러오기** 버튼이 표시됩니다.
이 버튼에서 내 수정을 버릴지 확인한 뒤 최신 라벨을 불러와 다시 수정·승인할 수 있습니다.
추론·학습 GPU 작업은 기존처럼 하나씩 실행됩니다.

접속이 안 되면 서버 컴퓨터의 방화벽에서 사용하는 TCP 포트(기본 `8765`)를 허용하고,
공유기의 게스트 Wi-Fi나 기기 간 통신 차단 설정을 확인하세요.
접속한 사람은 동일한 작업 권한을 사용하므로 함께 작업할 사람에게 주소를 공유하세요.
이 컴퓨터에서만 접속하게 실행하려면 다음을 사용합니다.

```bash
ANNOTATION_HOST=127.0.0.1 ./run.sh
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

### 외부에서 라벨링한 BBox 데이터 가져오기

YOLO ZIP의 `data.yaml`과 `train/`, `valid/`(또는 `val/`), `test/` 아래
`images/`·`labels/`를 함께 가져올 수 있습니다.

```bash
python -m bbox.import_yolo /path/to/dataset.zip --replace
```

기존 BBox 데이터는 `backups/bbox-dataset-날짜/`로 옮긴 뒤 교체합니다.
ZIP은 같은 위치의 확장자를 뺀 폴더에 풀고, 클래스 번호는 `data.yaml`의 이름을 기준으로
웹의 클래스 순서에 맞춰 변환합니다. 이미지·라벨과 원래 분할을 유지하고 **승인됨**으로 등록합니다.
ZIP의 `train`은 `bbox/dataset/train/`, `valid`는 `test/`(검증),
`test`는 `eval/`(최종 평가)에 들어갑니다. 데이터 버전을 다시 생성해도 가져온 분할을 유지합니다.
가져오기 이력과 클래스 변환 정보는 `bbox/dataset/_store/imports/`에 저장합니다.
작업할 미검수 이미지가 없으면 BBox 화면에서 승인된 이미지를 자동으로 보여줍니다.

## 포함 모델

- **로컬 BBox 모델:** `bbox/models/best.pt`가 있으면 웹의 추론·추가 학습 모델 목록에
  자동 등록하고 기본 선택합니다.
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
