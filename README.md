# 4D Gaussian Splatting (4DGS) Pipeline

완전 자동화된 4D Gaussian Splatting 파이프라인입니다. 멀티뷰 비디오나 이미지에서 4DGS 모델을 학습하고 렌더링할 수 있습니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd 4DGS

# Submodule 초기화
git submodule update --init --recursive

# 의존성 설치
pip install -r requirements.txt

# CUDA 확장 빌드 (필요시)
cd submodules/4dgs/submodules/depth-diff-gaussian-rasterization
pip install .
cd ../simple-knn
pip install .
```

### 2. 데이터셋 준비

#### 샘플 데이터셋 다운로드
```bash
./download_sample_data.sh sample_synthetic
```

#### 직접 데이터 준비
- **비디오 기반**: `data/inputs/your_dataset/videos/` 폴더에 비디오 파일 배치
- **이미지 기반**: `data/inputs/your_dataset/images/` 폴더에 이미지 파일 배치
- **카메라 포즈 포함**: `data/inputs/your_dataset/transforms.json` 파일 준비

### 3. 파이프라인 실행

#### 방법 1: Quick Start (가장 간단)
```bash
./quick_start.sh sample_synthetic
```

#### 방법 2: Full Pipeline Script
```bash
./run_pipeline.sh data/inputs/sample_synthetic my_experiment
```

#### 방법 3: Python 직접 실행
```bash
# 전체 파이프라인
python main.py --input data/inputs/sample_synthetic --experiment my_experiment

# 개별 단계
python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step preprocess
python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step colmap
python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step convert
python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step train
python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step render
```

## 📋 파이프라인 단계

1. **Preprocess**: 비디오에서 프레임 추출 또는 이미지 준비
2. **COLMAP**: 카메라 포즈 추정 (Structure-from-Motion)
3. **Convert**: COLMAP 출력을 `transforms.json` 형식으로 변환
4. **Train**: 4DGS 모델 학습
5. **Render**: 학습된 모델로 비디오 렌더링

## 🎯 주요 기능

### COLMAP 자동 스킵
`transforms.json` 파일이 있고 유효한 프레임이 포함되어 있으면 COLMAP 단계를 자동으로 건너뜁니다.

### 재개 (Resume) 지원
파이프라인이 중단되면 `--resume` 옵션으로 마지막 단계부터 재개할 수 있습니다:

```bash
./run_pipeline.sh data/inputs/sample_synthetic my_experiment --resume
```

### 커스텀 설정
YAML 설정 파일을 사용하여 파이프라인을 커스터마이즈할 수 있습니다:

```bash
./run_pipeline.sh data/inputs/sample_synthetic my_experiment --config configs/custom.yaml
```

## 📁 출력 구조

```
data/outputs/<experiment_name>/
├── model/
│   ├── point_cloud/iteration_30000/
│   │   └── point_cloud.ply          # 학습된 모델
│   ├── cfg_args                      # 학습 설정
│   └── training.log                  # 학습 로그
└── renders/
    └── render_interpolate.mp4        # 렌더링된 비디오
```

## 🎬 Inference 및 커스텀 렌더링

학습된 모델로 커스텀 카메라 경로를 렌더링하려면 `RENDER_GUIDE.md`를 참조하세요.

```bash
# 커스텀 카메라 경로 렌더링
python main.py \
    --input data/inputs/sample_synthetic \
    --experiment my_experiment \
    --step render \
    --render_path path/to/render_transforms.json \
    --output_dir custom_output
```

## 🔧 고급 옵션

### GPU 선택
```bash
./run_pipeline.sh data/inputs/sample_synthetic my_experiment --gpu 1
```

### 학습 반복 횟수 변경
```bash
./run_pipeline.sh data/inputs/sample_synthetic my_experiment --iterations 50000
```

### 출력 압축
```bash
./run_pipeline.sh data/inputs/sample_synthetic my_experiment --compress
```

## 📊 학습 모니터링

학습 중 진행 상황을 모니터링하려면:

```bash
# 별도 터미널에서
./monitor_training.sh <experiment_name>

# 또는 빠른 상태 확인
./check_status.sh <experiment_name>
```

## 🐛 문제 해결

### COLMAP 오류
- COLMAP이 설치되어 있는지 확인: `which colmap`
- 입력 이미지가 충분한지 확인 (최소 10-20장 권장)

### CUDA 오류
- CUDA 버전 확인: `nvcc --version`
- PyTorch CUDA 버전과 일치하는지 확인: `python -c "import torch; print(torch.version.cuda)"`

### 메모리 부족
- 배치 크기 줄이기 (설정 파일에서 `batch_size` 조정)
- 이미지 해상도 낮추기

## 📚 참고 자료

- [4DGaussians 원본 저장소](https://github.com/hustvl/4DGaussians)
- [RENDER_GUIDE.md](RENDER_GUIDE.md) - Inference 및 커스텀 렌더링 가이드

## 📝 라이선스

이 프로젝트는 원본 4DGaussians 저장소의 라이선스를 따릅니다.

