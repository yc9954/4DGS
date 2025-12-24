# 4DGS 렌더링 및 Inference 가이드

## 학습 완료 후 결과물 확인

### 1. 학습 결과물 위치
```
data/outputs/{experiment_name}/model/
  - training.log          # 학습 로그
  - point_cloud/          # 학습된 가우시안 포인트 클라우드
  - iteration_*/          # 체크포인트 (특정 iteration)
```

### 2. 렌더링 실행 방법

#### 방법 1: 파이프라인 스크립트 사용 (권장)
```bash
cd /workspace/4DGS
./run_pipeline.sh data/inputs/sample_synthetic sample_synthetic_test --resume
```
이미 학습이 완료되어 있으면 render 단계만 실행됩니다.

#### 방법 2: 렌더링만 직접 실행
```bash
cd /workspace/4DGS
python3 main.py \
  --input data/inputs/sample_synthetic \
  --experiment sample_synthetic_test \
  --step render \
  --skip_colmap
```

### 3. 렌더링 결과물 위치
```
data/outputs/{experiment_name}/renders/
  - frames/               # 렌더링된 프레임 이미지들
  - output.mp4           # 최종 비디오 (생성됨)
```

### 4. Inference 옵션

#### 렌더링 프레임 수 조정
```bash
python3 main.py \
  --input data/inputs/sample_synthetic \
  --experiment sample_synthetic_test \
  --step render \
  --render-frames 300 \
  --skip_colmap
```

#### 카메라 경로 타입 선택
- `interpolate`: 프레임 간 보간 (기본값)
- `spiral`: 나선형 경로
- `orbit`: 궤도 경로

```bash
python3 main.py \
  --input data/inputs/sample_synthetic \
  --experiment sample_synthetic_test \
  --step render \
  --render-path spiral \
  --skip_colmap
```

### 5. 결과물 확인

#### 비디오 파일 확인
```bash
ls -lh data/outputs/sample_synthetic_test/renders/*.mp4
```

#### 프레임 이미지 확인
```bash
ls data/outputs/sample_synthetic_test/renders/frames/
```

### 6. 결과물 다운로드

RunPod 환경에서는 `/workspace` 디렉토리가 마운트되어 있으므로:
- 웹 인터페이스에서 `/workspace/4DGS/data/outputs/` 경로로 접근
- 또는 압축하여 다운로드:
```bash
cd /workspace/4DGS
tar -czf results.tar.gz data/outputs/sample_synthetic_test/
```

