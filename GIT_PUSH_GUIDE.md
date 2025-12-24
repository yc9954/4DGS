# Git 푸시 가이드

## 현재 상태

변경사항이 스테이징되었습니다. 다음 단계를 따라 GitHub에 푸시하세요.

## 1. Git 사용자 정보 설정

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

또는 이 저장소에만 적용하려면:

```bash
git config user.email "your-email@example.com"
git config user.name "Your Name"
```

## 2. Submodule 확인

Submodule이 제대로 설정되어 있는지 확인:

```bash
# Submodule 상태 확인
git submodule status

# Submodule이 없으면 초기화
git submodule update --init --recursive
```

## 3. 변경사항 커밋

```bash
git commit -m "feat: Complete 4DGS pipeline with fixes and improvements

- Fix COLMAP skipping logic to check for valid frames in transforms.json
- Fix double file extension bug in dataset_readers.py
- Fix image data type conversion (np.byte -> np.uint8)
- Fix model saving path issues in scene/__init__.py
- Fix render.py to handle custom camera paths correctly
- Add PYTHONPATH setup for CUDA extensions in train/render wrappers
- Add error handling for network_gui port conflicts
- Ensure model saves at final iteration
- Add RENDER_GUIDE.md for inference instructions
- Add config files for sample datasets
- Add monitoring scripts (monitor_training.sh, check_status.sh)
- Add quick_start.sh and README.md
- Update run_pipeline.sh with improved error handling"
```

## 4. GitHub에 푸시

### 방법 1: HTTPS (토큰 필요)

```bash
# GitHub Personal Access Token이 필요합니다
# Settings > Developer settings > Personal access tokens > Tokens (classic)
git push origin claude/4d-gaussian-splatting-mvp-FNKzo
```

### 방법 2: SSH

```bash
# SSH 키가 설정되어 있다면
git remote set-url origin git@github.com:yc9954/4DGS.git
git push origin claude/4d-gaussian-splatting-mvp-FNKzo
```

## 5. Submodule 포함 여부

Submodule의 변경사항도 함께 푸시하려면:

```bash
# Submodule로 이동하여 커밋
cd submodules/4dgs
git add .
git commit -m "Update 4DGS submodule"
git push

# 원래 저장소로 돌아와서 submodule 참조 업데이트
cd ../..
git add submodules/4dgs
git commit -m "Update submodule reference"
git push
```

## 현재 스테이징된 파일

- `main.py` - 메인 파이프라인 스크립트
- `run_pipeline.sh` - 실행 스크립트
- `src/render_wrapper.py` - 렌더링 래퍼
- `src/train_wrapper.py` - 학습 래퍼
- `setup_runpod.sh` - RunPod 설정
- `RENDER_GUIDE.md` - 렌더링 가이드
- `README.md` - 프로젝트 README
- `quick_start.sh` - 빠른 시작 스크립트
- `monitor_training.sh` - 학습 모니터링 스크립트
- `check_status.sh` - 상태 확인 스크립트
- `configs/` - 설정 파일들
- `.pipeline_state/` - 파이프라인 상태 파일

