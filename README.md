# VisionInference

TensorRT 기반 비전 AI 검사 파이프라인을 개발하는 Windows C++20 DLL 프로젝트입니다. Python에서 변환한 YOLO26 모델을 로드하고, 이미지 전처리·추론·후처리를 제공하는 라이브러리를 목표로 합니다.

현재 개발 진행 중입니다. C ABI와 런타임 생성·종료, Gray8 이미지 메타데이터 검증을 구현한 초기 단계이며, 모델 로드와 실제 추론은 아직 구현하지 않았습니다. Classification부터 시작해 Detection과 Instance Segmentation으로 확장할 계획입니다.

전체 이미지 Resize + Letterbox와 원본 배율 1:1 타일링, 전처리·후처리의 독립적인 CPU/GPU 선택, 배치 처리는 개발 예정입니다. TensorRT 추론 자체는 GPU에서 수행하며, 목표 입력 형식은 Gray8/BGR8/RGB8, 최대 크기는 30,000 × 30,000입니다. 실제 지원 범위는 메모리 예산과 구현 단계에 따라 검증할 예정입니다.

## 빌드 환경

- Windows 11 x64
- Visual Studio 2026, C++ 데스크톱 개발 도구 및 Windows SDK
- MSVC v145, C++20
- Debug: `/MDd`, Release: `/MD`

현재 초기 DLL 코드는 CUDA/TensorRT 없이 빌드하도록 구성되어 있습니다. 추론 구현에 필요한 NVIDIA GPU, CUDA Toolkit, TensorRT의 버전 조합과 연결 설정은 개발 진행 중이며 아직 확정하지 않았습니다.

## 폴더 구조

```text
VisionInference/
├─ VisionInference.sln
├─ VisionInference/
│  ├─ VisionInference.vcxproj
│  ├─ VisionInference.h       # 공개 C ABI
│  └─ VisionInference.cpp     # 초기 런타임 및 입력 검증
├─ docs/                     # 개발 계획 및 최적화 참고 자료 (로컬 관리)
└─ x64/                      # 로컬 빌드 산출물
```

현재 솔루션은 VisionInference DLL 프로젝트 하나로 구성됩니다. 클래스별 내부 구현과 Python 모델 변환·검증 스크립트는 개발 진행 중입니다.

## 저장소 배치 및 의존성

현재 프로젝트에는 Core, CUDA, TensorRT 의존성이 연결되어 있지 않습니다. 모델 변환은 외부 Python 환경에서 수행하고, C++ DLL에서는 변환된 모델을 여는 방식으로 개발할 예정입니다. 모델 파일 형식과 메타데이터 규약은 개발 진행 중입니다.

Core 이미지 타입 재사용을 검토하는 폴더 배치는 다음과 같습니다.

```text
Module/
├─ Core/
│  ├─ ImageType/
│  │  ├─ ImageTypeBase.h
│  │  ├─ ImageView.h
│  │  └─ Image.h
│  └─ ShapeType/
└─ VisionInference/
   ├─ VisionInference.sln
   └─ VisionInference/
```

Core의 이미지 타입은 헤더 기반입니다. 도입 시 내부 C++ 코드에서 include 경로로 참조하고, 외부 C ABI에는 포인터·크기·stride 등으로 구성된 입력 구조체를 유지하는 방향을 검토 중입니다. 이미지 타입만 사용할 경우 Core 정적 라이브러리 링크는 필요하지 않습니다. 실제 프로젝트 참조 설정은 아직 적용하지 않았습니다.

모델 가중치, 변환된 엔진, 로컬 검사 데이터 및 외부 SDK 바이너리는 Git 관리 대상에서 제외합니다. 모델·데이터는 각각 루트 `models/`, `data/` 폴더에서 로컬 관리할 수 있습니다.

## 빌드 및 실행

1. Visual Studio에서 `VisionInference.sln`을 엽니다.
2. Debug 또는 Release 구성과 x64 플랫폼을 선택합니다.
3. VisionInference 프로젝트를 빌드합니다.
4. 생성된 DLL은 추후 C++ 테스트 프로그램 또는 C# WPF 프로그램에서 공개 C ABI를 통해 호출합니다.

DLL 프로젝트이므로 단독 실행 프로그램은 없습니다. 솔루션 기준 기본 출력 위치는 `x64/Debug/` 또는 `x64/Release/`이며, 실제 위치는 Visual Studio의 출력 디렉터리 설정을 확인합니다. 소비 프로젝트에는 `VISIONINFERENCE_EXPORTS`를 정의하지 않습니다.

모델 로드·추론 실행 예제, 테스트 실행 방법과 성능 측정 결과는 개발 진행 중입니다.
