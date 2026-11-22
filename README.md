# Annotation Tool

BBox와 차선 Segmentation을 편집·승인하고 모델을 추가 학습하는 로컬 웹 도구입니다.

## 설치

Python 3.10 이상을 사용합니다. GPU를 사용하려면 환경에 맞는 PyTorch/CUDA를 먼저 설치하세요.

```bash
git clone https://github.com/Minjea31/annotation_tool.git
cd annotation_tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 혼자 사용하기

설치한 컴퓨터의 터미널에서 서버를 실행합니다.

```bash
source .venv/bin/activate
ANNOTATION_HOST=127.0.0.1 ./run.sh
```

1. 브라우저에서 **http://127.0.0.1:8765**에 접속해 **BBox** 또는 **Segmentation**을 선택합니다.
2. **＋ 데이터셋 추가**에서 이 컴퓨터에 있는 이미지 폴더의 절대 경로를 입력합니다.
   이미지를 관리 폴더에 복사하며 이미 등록한 이미지는 건너뜁니다.
3. 모델을 선택해 추론한 다음 박스·마스크를 수정합니다. 직접 라벨을 작성해도 됩니다.
4. **저장**하면 편집본을 보관하고, **승인·다음**을 누르면 저장·승인 후 다음 이미지로 이동합니다.
   미완성 이미지는 **재추론 대기**, 사용하지 않을 이미지는 **제외**로 분류합니다.
5. 저장·승인한 이미지가 최소 5장 모이면 데이터셋 생성과 추가 학습을 실행할 수 있습니다.
   학습 중에도 라벨링할 수 있으며 GPU 작업은 순서대로 실행됩니다.

종료할 때 수정 내용을 저장한 뒤 서버 터미널에서 `Ctrl+C`를 누릅니다.
다음에 같은 설정으로 실행하면 기존 이미지·라벨·검수 이력을 이어서 사용합니다.
`launcher.html`을 직접 열어도 서버를 먼저 실행해야 합니다.

`./run.sh`는 저장소의 `.venv`, 현재 Python, 기존 Conda `yolo` 환경 순서로
필요한 패키지가 있는 Python을 찾습니다. Python이나 포트를 지정하려면 다음처럼 실행합니다.

```bash
ANNOTATION_PYTHON=/path/to/python ANNOTATION_HOST=127.0.0.1 ANNOTATION_PORT=8766 ./run.sh
```

## 여러 명이 함께 사용하기

### 서버 담당자

이미지와 모델을 보관할 컴퓨터 **한 대에서 서버 하나만** 실행합니다.

```bash
source .venv/bin/activate
./run.sh
# 기본값: ANNOTATION_HOST=0.0.0.0, ANNOTATION_PORT=8765
```

터미널에 표시되는 **같은 Wi-Fi에서 접속** 주소(예: `http://192.168.0.10:8765`)를
작업자에게 공유합니다. 서버 컴퓨터에서는 `http://127.0.0.1:8765`로도 접속할 수 있습니다.
작업하는 동안 서버와 컴퓨터를 켜 두고 절전 모드를 해제합니다.
기존 서버에 코드 변경을 적용하려면 작업자들이 저장한 뒤 서버를 종료하고 다시 실행합니다.

### 함께 작업하는 사람

1. 서버와 같은 Wi-Fi 또는 서로 통신 가능한 유선 LAN에 연결합니다.
2. 공유받은 주소를 브라우저에서 엽니다. Python 설치나 저장소 복제는 필요하지 않습니다.
3. BBox 또는 Segmentation을 선택하고 폴더·세션·이미지 구간을 나눠 작업합니다.
4. 수정 후 **저장**하거나 **승인·다음**을 누릅니다. 다른 사람이 저장한 상태는
   BBox 약 2.5초, Segmentation 약 3초 간격으로 자동 갱신됩니다.
   숨긴 탭은 주기 갱신을 쉬고 다시 열면 즉시 갱신합니다.

모든 저장·승인·삭제·학습은 **서버 컴퓨터**에서 처리됩니다.
**데이터셋 추가에 입력하는 경로도 서버 컴퓨터의 경로**입니다.
다른 작업자의 PC에만 있는 이미지는 먼저 서버 컴퓨터로 옮긴 뒤 추가합니다.
서버를 여러 대 켜거나 같은 데이터 폴더에 여러 서버·여러 worker를 연결하지 마세요.
동시 편집 보호와 GPU 대기는 단일 서버 프로세스에서 관리합니다.

### 같은 이미지를 수정했을 때

서로 다른 이미지를 나눠 작업하는 것을 권장합니다. 같은 이미지에서 오래된 편집본을
저장하면 **변경 충돌**로 거절되어 먼저 저장된 라벨을 덮어쓰지 않습니다.
편집하지 않는 이미지는 다른 사람이 저장한 최신 라벨을 자동으로 불러옵니다.
Segmentation에서 충돌이 나면 내 수정은 유지되고 **최신 라벨 불러오기** 버튼이 표시됩니다.
내 수정을 버릴지 확인한 뒤 최신 라벨을 불러와 다시 수정·승인합니다.
BBox에서는 충돌 안내가 나오면 현재 이미지의 최신 라벨을 다시 열어 확인합니다.

### 접속·속도 확인

접속이 안 되면 서버 컴퓨터의 방화벽에서 TCP 포트(기본 `8765`)를 허용하고,
공유기의 게스트 Wi-Fi나 기기 간 통신 차단 설정을 확인합니다.
접속한 사람은 동일한 작업 권한을 사용하므로 함께 작업할 사람에게 주소를 공유하세요.
이미지·DB는 서버의 로컬 SSD에 두고, 가능하면 서버를 유선 LAN으로 연결합니다.
데이터 위치는 다음처럼 지정할 수 있습니다. 지정한 경로의 데이터로 열리므로
기존 작업을 옮길 때는 서버를 종료한 뒤 데이터 폴더 전체를 먼저 복사하세요.

```bash
BBOX_REVIEW_DATA=/data/annotation/bbox/dataset \
ANNOTATION_SEG_DATA=/data/annotation/segmentation/dataset ./run.sh
```

저장 시 변경되지 않은 이미지·라벨은 다시 복사하지 않고, 분할이 바뀌거나 수정된
파일만 연결합니다. BBox 승인도 저장과 함께 한 번의 요청으로 처리합니다.
전체 목록은 서버에서 재사용하며 변경이 없는 갱신은 본문 없는 응답(HTTP 304)으로
처리해 전송과 화면 재생성을 줄입니다. SQLite WAL과 조회 인덱스로 동시 조회·저장의
대기 시간을 줄였으며 저장 완료 응답 전에 라벨과 데이터 폴더 반영을 마칩니다.

## 폴더 구조

```text
bbox/
  app.py, engine.py, static/  # 박스 편집·추론·학습
  dataset/                   # BBox 데이터 및 검수 이력 (로컬)
  models/                    # 최신 YOLO26 Nano 가중치와 model.json
  vendor/                    # 동봉한 Ultralytics
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

모든 위치의 `dataset/` 내용과 `datasets/`, 검수 DB, 과거 학습 결과와 캐시는 Git에서 제외합니다.
빈 데이터 폴더를 표시하는 `.gitkeep`만 버전 관리합니다.
새로 복제하면 데이터는 비어 있으며 처음 실행할 때 관리 폴더가 생성됩니다.
백업할 때는 각 도구의 `dataset/`과 `models/`를 함께 보관하세요.
데이터 위치는 `BBOX_REVIEW_DATA`, `ANNOTATION_SEG_DATA` 환경변수로 변경할 수 있습니다.

## 편집 단축키와 데이터 가져오기

BBox 주요 단축키: `D` 박스 추가, `A` 선택·이동, `S`/`Delete` 박스 삭제,
`1`~`8` 클래스 변경, `Z`/`X` 실행 취소·다시 실행, `V` 저장, `W` 승인·다음, `Q`/`E` 이전·다음.
휠로 확대하고 Space+드래그로 이동합니다.
클래스 순서는 `slow, bump, crosswalk, left, right, parking, red, green`입니다.

Segmentation 주요 단축키: `A` 브러시, `S` 지우개, `D` 폴리곤, `F` 폴리곤 완성,
`Esc` 폴리곤 취소, `1` 주황 차선, `2` 흰 점선, `Z`/`X` 실행 취소·다시 실행,
`V` 저장, `W` 승인·다음, `R` 재추론 대기, `Q`/`E` 이전·다음, `C` 마스크 표시 전환.
휠로 확대하고 Space+드래그로 이동하며, Shift+휠로 브러시 크기를 조절합니다.

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

각 도구에 가장 최근 완료된 학습의 모델 하나씩만 남겼습니다.

- **BBox:** `bbox/models/best.pt` — 2026-11-18 학습한 YOLO26 Nano,
  `slow, bump, crosswalk, left, right, parking, red, green` 8개 클래스.
- **Segmentation:** `segmentation/models/best.pt` — 2026-11-19 마지막 완료 학습
  (`7f5bd8`)의 best 체크포인트, MobileNetV3 U-Net.
  검증 차선 IoU 약 95.48%, 최종 eval 차선 IoU 약 94.84%입니다.
  마스크 값은 `0` 배경, `1` orange_lane, `2` white_dashed_lane, `255` 무시입니다.

출처와 SHA-256은 각 `models/model.json`에 기록했습니다. 이전 학습의 best/last,
중복 내보내기와 사전 학습 캐시는 정리했습니다. 새 학습을 실행하면 실행 폴더에
새 체크포인트가 생성됩니다. ImageNet 초기 가중치로 새로 학습할 때는 필요한 캐시를
다시 내려받습니다.

새로 복제한 환경의 Segmentation 추가 학습은 **처음부터 학습**으로 시작합니다.
기존 검수 DB가 있는 환경에서는 같은 데이터 정책에 연결된 모델로 이어 학습할 수 있습니다.
이미지·라벨·검수 이력은 `dataset/`에 보관하며 Git에서 제외합니다.

## 검증

```bash
OMP_NUM_THREADS=4 python -m unittest bbox.test_bbox bbox.test_yolo_import test_performance -v
OMP_NUM_THREADS=4 python -m unittest discover -s segmentation/train -p test_training.py -v
node --test bbox/test_inference_display.cjs
```

Google Chrome(`/usr/bin/google-chrome`)이 설치된 환경에서는 실제 두 브라우저의
공동 작업, 자동 갱신, 변경 충돌과 승인 요청도 검증할 수 있습니다.

```bash
python -m pip install -r segmentation/review/requirements-test.txt
OMP_NUM_THREADS=4 python -m unittest discover -s segmentation/review -p test_collaboration.py -v
```
