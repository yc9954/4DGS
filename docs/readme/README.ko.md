<h1 align="center">4DGS Pipeline</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-4493F8?style=flat" alt="Python 3" />
  <img src="https://img.shields.io/badge/PyTorch%20%2B%20CUDA-hustvl%2F4DGaussians-4493F8?style=flat" alt="PyTorch + CUDA, hustvl/4DGaussians" />
  <img src="https://img.shields.io/badge/COLMAP%20%C2%B7%20FFmpeg-4493F8?style=flat" alt="COLMAP and FFmpeg" />
  <img src="https://img.shields.io/badge/target-RunPod-4493F8?style=flat" alt="Target: RunPod" />
</p>

<p align="center">
  <sub><a href="../../README.md">English</a></sub>
</p>

<p align="center">
  <strong>멀티뷰 비디오를 넣으면 학습된 4D Gaussian Splatting 모델과 렌더링 영상이 나옵니다.</strong><br/>
  프레임 추출, COLMAP 포즈 추정, <code>transforms.json</code> 변환, <a href="https://github.com/hustvl/4DGaussians">hustvl/4DGaussians</a> 학습,<br/>
  보간·나선·궤도 카메라 경로 렌더링까지 명령 하나로 실행하며, 단계별 재개와 YAML 설정, RunPod 설치 스크립트를 제공합니다.
</p>

<h3 align="center"><a href="#시작하기"><ins>시작하기</ins></a></h3>

## 주요 기능

- **다섯 단계, 하나의 CLI.** `main.py`가 `preprocess → colmap → convert → train → render`를 순서대로 실행합니다. `--step`으로 일부만 고르거나, `run_pipeline.sh --resume`으로 실패한 지점부터 재개하고(`.pipeline_state/<experiment>.state`에 상태 저장), `--no-skip`으로 전부 다시 돌릴 수 있습니다.
- **FFmpeg 프레임 추출.** `src/preprocess.py`가 각 비디오를 분석해 지정한 `fps`, 포맷, 선택적 리사이즈로 동기화된 프레임을 카메라별 폴더에 저장합니다.
- **헤드리스 COLMAP.** `src/colmap_wrapper.py`가 특징 추출, exhaustive/sequential 매칭, sparse 재구성을 Python에서 GUI 없이 실행하고, `src/convert_colmap.py`가 바이너리·텍스트 모델을 NeRF 스타일 `transforms.json`(내부 파라미터, 이미지별 4x4 포즈)으로 변환합니다.
- **COLMAP 자동 스킵과 공개 데이터셋.** 유효한 `transforms.json`이 이미 있으면 COLMAP을 건너뜁니다. `DatasetAdapter`가 N3DV, D-NeRF, LLFF 레이아웃을 감지해 파이프라인 형식으로 재배치하고, `download_sample_data.sh`로 N3DV `coffee` / `flame_salmon`, D-NeRF `lego`를 받거나 작은 `sample_synthetic` 장면을 생성할 수 있습니다.
- **학습은 업스트림 코드에 위임.** `src/train_wrapper.py`가 `submodules/4dgs/train.py`(git submodule로 고정한 hustvl/4DGaussians)를 찾아 iteration, 저장·테스트 iteration, 해상도를 넘겨 실행하고 로그를 `training.log`로 남깁니다. `--mock`은 가짜 체크포인트를 만들어 GPU 없이 나머지 파이프라인을 점검하게 해 줍니다.
- **폴백 체인이 있는 렌더링.** `src/render_wrapper.py`가 카메라 경로(학습 포즈 사이 `interpolate`, `spiral`, `orbit`)를 만들고, 업스트림 `render.py`가 있으면 그것을, 없으면 순수 NumPy 구현인 `src/gaussian_renderer.py`(비등방성 2D 가우시안, 알파 합성, 구면 조화 함수)를, 그것도 실패하면 플레이스홀더 프레임을 써서 비디오 단계가 끝나도록 합니다. 인코딩은 FFmpeg(`libx264`, `crf 18`)입니다.
- **인터랙티브 4D 뷰어.** `src/viewer_4d.py`(pygame)로 학습된 모델을 공간과 시간 양쪽으로 탐색합니다. WASD/QE와 마우스로 이동, 방향키와 스페이스로 타임라인 이동·재생, `1`-`9`로 점프. `./view_4d.sh <experiment>`로 실행합니다.

**함께 포함된 것**

- `setup_runpod.sh`: 시스템 패키지, FFmpeg, COLMAP 설치, 컨테이너의 CUDA(12.1 / 11.8 / 11.7 / 11.6)에 맞는 PyTorch 선택, requirements 설치, submodule 초기화, `depth-diff-gaussian-rasterization`과 `simple-knn` CUDA 확장 빌드.
- 별도 터미널에서 진행 상황을 보는 `monitor_training.sh`, `check_status.sh`; 클라우드에서 내려받기 위한 `--compress`.
- 렌더링 옵션과 RunPod에서 결과 가져오기를 다룬 `RENDER_GUIDE.md`, 컨테이너에서 푸시하는 방법을 다룬 `GIT_PUSH_GUIDE.md`.

---

## 동작 방식

```text
data/inputs/<dataset>/videos/*.mp4  (또는 images/, 또는 포즈가 포함된 데이터셋)
        │
        ▼  preprocess   FFmpeg 프레임 추출  ──▶ data/processed/<exp>/images/
        ▼  colmap       feature_extractor → matcher → mapper   (transforms.json이 있으면 스킵)
        ▼  convert      COLMAP sparse 모델 ──▶ data/outputs/<exp>/transforms.json
        ▼  train        submodules/4dgs/train.py -s <source> -m <model> --iterations N
        │               ──▶ data/outputs/<exp>/model/point_cloud/iteration_N/point_cloud.ply
        ▼  render       카메라 경로 (interpolate | spiral | orbit)
                        ──▶ 업스트림 render.py  ▷ 네이티브 NumPy 렌더러  ▷ 플레이스홀더 프레임
                        ──▶ FFmpeg ──▶ data/outputs/<exp>/renders/*.mp4
```

1. **설정.** `PipelineConfig`(`src/config.py`)가 기본값, 선택적 `--config` YAML(`configs/` 참고), CLI 플래그를 병합합니다. `skip_existing: true`로 모든 단계가 멱등하게 동작합니다.
2. **적응과 전처리.** 공개 데이터셋은 먼저 재배치되고, 비디오는 카메라별 폴더의 프레임으로 나뉩니다.
3. **포즈.** 설정된 카메라 모델(기본 `OPENCV`)과 매처로 COLMAP을 실행하고 결과를 `transforms.json`으로 변환하며, 학습기가 기대하는 위치에 이미지를 심볼릭 링크합니다.
4. **학습.** 래퍼가 체크포인트 존재를 확인하고 렌더링 시 소스 경로와 iteration을 찾을 수 있도록 `cfg_args`를 함께 저장합니다.
5. **렌더링과 인코딩.** 최신 `iteration_*`이 자동 선택되며 `--render-frames`로 길이를 정합니다.

<details>
<summary><strong>출력 구조</strong></summary>

```text
data/outputs/<experiment>/
├── transforms.json
├── model/
│   ├── point_cloud/iteration_30000/point_cloud.ply
│   ├── cfg_args
│   └── training.log
└── renders/
    ├── frames/
    └── render_interpolate.mp4
```

</details>

---

## 기술 스택

<p>
  <kbd>Python&nbsp;3</kbd> &nbsp; <kbd>PyTorch&nbsp;1.13&nbsp;–&nbsp;2.1&nbsp;(CUDA)</kbd> &nbsp; <kbd>hustvl/4DGaussians</kbd> &nbsp; <kbd>mmcv&nbsp;1.6</kbd> &nbsp; <kbd>COLMAP</kbd> &nbsp; <kbd>FFmpeg</kbd> &nbsp;
  <kbd>NumPy&nbsp;&lt;2</kbd> &nbsp; <kbd>OpenCV</kbd> &nbsp; <kbd>open3d</kbd> &nbsp; <kbd>plyfile</kbd> &nbsp; <kbd>PyYAML&nbsp;/&nbsp;OmegaConf</kbd> &nbsp; <kbd>loguru</kbd> &nbsp; <kbd>pygame</kbd> &nbsp; <kbd>pytest</kbd>
</p>

---

## 시작하기

**사전 요구 사항**

- NVIDIA GPU가 있는 Linux, 그리고 PyTorch 빌드와 맞는 CUDA 툴킷(업스트림 CUDA 확장을 설치 시 컴파일합니다).
- `PATH`에 COLMAP과 FFmpeg (`setup_runpod.sh`가 apt로 둘 다 설치합니다).
- pip가 있는 Python 3. NumPy는 2.0 미만이어야 합니다.

```bash
git clone https://github.com/yc9954/4DGS.git
cd 4DGS
git submodule update --init --recursive        # hustvl/4DGaussians를 submodules/4dgs로 받음

# RunPod / Ubuntu 컨테이너: 한 번에
chmod +x setup_runpod.sh && ./setup_runpod.sh

# 수동: 먼저 CUDA에 맞는 torch를 설치한 뒤(requirements.txt 상단 참고)
pip install -r requirements.txt
cd submodules/4dgs/submodules/depth-diff-gaussian-rasterization && pip install . && cd ../simple-knn && pip install . && cd ../../../..
```

**실행**

```bash
./download_sample_data.sh sample_synthetic          # 또는: coffee, flame_salmon, lego, --list
./quick_start.sh sample_synthetic                    # 가장 간단한 경로

./run_pipeline.sh data/inputs/sample_synthetic my_experiment            # 전체 파이프라인
./run_pipeline.sh data/inputs/sample_synthetic my_experiment --resume   # 실패 후 이어서
./run_pipeline.sh data/inputs/coffee coffee --skip_colmap --config configs/coffee.yaml

python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step train --step render
python main.py --input data/inputs/sample_synthetic --experiment my_experiment --step render --render-path spiral --render-frames 300
./view_4d.sh my_experiment --time 0.5
```

직접 준비한 데이터는 `data/inputs/<name>/videos/`(멀티뷰 비디오), `data/inputs/<name>/images/`, 또는 준비된 `transforms.json`과 함께 두면 됩니다.

| 플래그 / 스크립트 옵션 | 기본값 | 역할 |
| --- | --- | --- |
| `--step` | 다섯 단계 전부 | `preprocess`, `colmap`, `convert`, `train`, `render` 중 하나 이상. |
| `--config` | — | `configs/default.yaml`을 덮어쓰는 YAML. `download_sample_data.sh`가 데이터셋별로 하나씩 생성. |
| `--iterations`, `--resolution` | `30000`, `-1` | 학습 길이와 입력 해상도(`-1` = 원본). |
| `--matcher`, `--no-gpu` | `exhaustive`, GPU 사용 | COLMAP 매처(비디오는 `sequential`)와 GPU 토글. |
| `--skip_colmap`, `--adapt_dataset` | 꺼짐, `auto` | 포즈 추정 생략; `n3dv` / `dnerf` / `llff` / `none` 강제. |
| `--render-path`, `--render-frames` | `interpolate`, `300` | 카메라 경로 종류와 길이. |
| `--method`, `--method_path` | `4dgs` | 학습 엔진(`4dgs`, `4dgaussians`, `fudan`, `custom`)과 `train.py` 위치. |
| `--mock`, `--compress`, `--no-skip`, `-v` | 꺼짐 | 드라이런용 가짜 학습; 출력 tar 압축; 기존 출력 무시; 상세 로그. |
| `run_pipeline.sh --gpu N` | `0` | 전체 실행에 사용할 CUDA 장치. |

---

## 빌드와 테스트

```bash
pytest tests/            # src/config.py, src/convert_colmap.py 단위 테스트
python main.py --input data/inputs/sample_synthetic --experiment dry --mock    # GPU 없이 파이프라인 점검
```

CI 워크플로는 없습니다. 테스트는 설정 기본값과 COLMAP → `transforms.json` 변환(카메라 모델, 4x4 행렬의 회전과 역행렬)을 다루며, 학습·렌더링 래퍼는 실제 실행으로만 확인됩니다.

---

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| `main.py` | 다섯 단계, 데이터셋 적응, COLMAP 자동 스킵, CLI를 가진 `Pipeline` 클래스. |
| `src/config.py` | 데이터클래스 설정(프레임 추출, COLMAP, 학습, 렌더)과 YAML 로딩. |
| `src/preprocess.py` | FFmpeg 프레임 추출, `VideoInfo`, `DatasetAdapter`(N3DV, D-NeRF, LLFF). |
| `src/colmap_wrapper.py`, `src/convert_colmap.py` | 헤드리스 SfM과 `transforms.json` 변환기. |
| `src/train_wrapper.py` | 업스트림 `train.py` 탐색·실행, mock 모드, 체크포인트 검증과 `cfg_args`. |
| `src/render_wrapper.py`, `src/gaussian_renderer.py` | 카메라 경로, 업스트림 / 네이티브 / 플레이스홀더 렌더링, FFmpeg 인코딩. |
| `src/viewer_4d.py`, `view_4d.sh` | 인터랙티브 pygame 뷰어. |
| `configs/` | `default.yaml`과 생성된 `sample_synthetic.yaml`, `coffee.yaml`, `test_sample.yaml`. |
| `run_pipeline.sh`, `quick_start.sh`, `monitor_training.sh`, `check_status.sh` | 재개, GPU 선택, 모니터링을 갖춘 셸 프런트엔드. |
| `setup_runpod.sh`, `download_sample_data.sh` | 환경과 데이터셋 부트스트랩. |
| `submodules/4dgs` | git submodule → `hustvl/4DGaussians`(`master`). |
| `tests/` | pytest 스위트. |
| `data/{inputs,processed,outputs}/`, `logs/`, `.pipeline_state/` | 런타임 디렉터리(`.gitkeep` 외 git-ignore). 기록된 상태 파일 하나는 `sample_synthetic_test` 실행이 `render`에서 실패했음을 보여줍니다. |
| `RENDER_GUIDE.md`, `GIT_PUSH_GUIDE.md` | 렌더링/추론과 RunPod에서 푸시하기 가이드. |

---

## 프로젝트 상태

**현재 동작.** 설정, 프레임 추출, 헤드리스 COLMAP, `transforms.json` 변환(단위 테스트 있음), 데이터셋 어댑터, 샘플 데이터 다운로드, submodule 기반 학습과 체크포인트 검증, 카메라 경로 생성, FFmpeg 인코딩, 재개 상태, RunPod 설치 스크립트, pygame 뷰어. 2025년 12월 말까지의 커밋 이력에 RunPod에서 실행하며 찾은 수정(sudo 없는 root 컨테이너, 학습용 이미지 심볼릭 링크, `cfg_args` 저장, `render_wrapper.py` 재작성)이 기록되어 있습니다.

**알아둘 폴백.** 업스트림 `render.py`를 찾지 못하면 NumPy 스플래터로 렌더링하는데 느리고 근사치입니다. 그것마저 실패하면 플레이스홀더 프레임을 써서 `.mp4`는 만들어집니다. 영상을 믿기 전에 `renders/frames/`를 확인하세요. `--mock`은 아주 작은 가짜 `point_cloud.ply`를 씁니다.

**알려진 한계.** 마지막으로 기록된 `sample_synthetic_test` 실행은 render 단계에서 실패했습니다(`.pipeline_state/`). `--method fudan`과 `custom`은 선언만 되어 있고 submodule로 고정된 것은 hustvl 레이아웃뿐입니다. 결과, 지표, 이미지는 공개된 것이 없으며 이 README의 어떤 내용도 벤치마크로 읽어서는 안 됩니다.

**크레딧.** 학습과 CUDA 래스터라이저는 [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians)(4D Gaussian Splatting for Real-Time Dynamic Scene Rendering)입니다. 네이티브 렌더러는 Kerbl et al., 3D Gaussian Splatting(2023)을 따르며 Blender의 `gpu_shader_3D_point_varying_size_varying_color`에서 포인트 렌더링 아이디어를 빌렸습니다. 샘플 데이터셋: [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video), [D-NeRF](https://github.com/albertpumarola/D-NeRF).

---

## 라이선스

아직 LICENSE 파일이 커밋되어 있지 않으므로 기본 저작권이 적용됩니다: all rights reserved. 이전 README는 이 프로젝트가 업스트림 4DGaussians 저장소의 라이선스를 따른다고 밝혔습니다. submodule은 자체 라이선스를 유지하며, 래퍼 코드에 대한 그 의도는 여기 그대로 남겨 둡니다.
