# Windows 개발·빌드

Python 3.12, Git, uv를 사용합니다. 단위 검사는 번역 모델/GPU를 사용하지 않습니다.
OCR 검사는 의존성에 포함된 소형 모델을 CPU에서 실행하며 Windows Arial 글꼴로 합성 이미지를 만듭니다.

```powershell
uv sync --frozen --group dev
uv run pytest -q
uv run python scripts/audit_public.py
```

`audit_public.py`는 Git index의 실제 파일 내용을 검사하므로 변경 사항을 stage한 뒤 실행하세요.
로컬 경로·일부 인증키 형식·생성/바이너리 파일과 공개 파일 허용 목록을 검사합니다.
모든 민감 정보를 탐지한다는 보증은 아니므로 diff도 함께 확인하세요.

## GPU 통합 시험

모델과 런타임 설치 후 실행합니다. 앱을 사용하는 데이터 루트와 분리한 `PAPERTRANSLATOR_HOME`을 권장합니다.
시험은 프로젝트의 `test_vault`와 합성 PDF만 사용합니다.

```powershell
uv run python scripts/integration.py
uv run python scripts/ocr_integration.py
uv run python scripts/recovery_checks.py
uv run python scripts/visual_validate.py
```

시각 검증은 별도로 설치된 Poppler의 `pdftoppm`이 PATH에 있어야 합니다.
PNG를 생성한 뒤 실제 페이지의 글꼴·잘림·숫자와 대조본을 직접 확인하세요.

## 실행 파일과 배포 파일

```powershell
uv run python -m PyInstaller --version
./scripts/build.ps1
uv run python scripts/collect_notices.py
uv run python scripts/package_public.py
```

빌드는 프로세스 내부 PATH만 격리해 다른 앱의 DLL이 섞이지 않게 합니다.
배포 패키지는 Git의 커밋된 소스 아카이브와 빌드된 앱, 의존 라이선스, SHA256을 생성합니다.
작업 기록·번역 모델·추론 런타임은 복사하지 않습니다. 소형 OCR 모델은 포함합니다. 실행 파일은 서명되지 않은 알파입니다.
앱 폴더의 Qt DLL을 별도 파일로 유지하고, 라이선스·재빌드 경로를 함께 배포하세요.
OCR 모델과 config.yaml은 PyInstaller의 collect-data로 포함합니다. 빌드 후 `scripts/ocr_integration.py --exe dist/PaperTranslator/PaperTranslator.exe`로 실제 OCR/번역 worker를 검사하세요.

실행 파일 사용자도 첫 사용 전에 이 소스 폴더의 `scripts/bootstrap.py`를 실행해 모델과 런타임을 설치해야 합니다.
모델 다운로드 GUI와 완전 자동 설치 프로그램은 아직 없습니다.

## CI

Windows CI는 Python 3.12 단위 검사와 공개 파일 검사를 실행합니다.
수동 실행은 Windows 앱을 빌드해 아티팩트로 올립니다. CI에서는 GPU/의미 정확도 검사를 수행하지 않습니다.
