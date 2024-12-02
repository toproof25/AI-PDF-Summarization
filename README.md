# AI-PDF-Summarization

---
# dataset
- [AI Hub 요약문 및 레포트 생성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=582)

# KoBART-summarization
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)



---
# `create_dataset_to_tsv.ipynb`
- 데이터셋을 tsv로 추출하는 코드 (jupyter notebook)

## JSON 데이터 처리 및 TSV 파일 생성

이 프로젝트는 여러 폴더에 저장된 JSON 데이터를 읽고, 학습용 및 시험용 데이터를 처리하여 TSV 파일로 변환하는 파이썬 스크립트입니다.

1. **폴더 목록 지정**: 다양한 종류의 데이터를 포함하는 여러 폴더를 설정합니다.
2. **JSON 데이터 읽기**: 각 폴더에 있는 JSON 파일을 읽고, 필요한 데이터를 추출하여 리스트에 저장합니다.
3. **데이터 프레임 생성**: 추출한 데이터를 `pandas` DataFrame으로 변환합니다.
4. **TSV 파일로 저장**: 처리된 데이터를 TSV 형식으로 저장하여 외부 시스템에서 사용할 수 있도록 합니다.

## 사용 방법

### 1. `folder_list`에 있는 폴더들에서 JSON 파일을 읽어들입니다.
각 폴더는 아래와 같은 형태로 구성되어 있으며, 각 JSON 파일은 두 가지 주요 항목을 포함합니다:
- `Meta(Refine)` → `passage`: 뉴스 본문
- `Annotation` → `summary1`: 뉴스 요약

### 2. 데이터를 TSV 형식으로 변환합니다.
학습 데이터와 시험 데이터를 각각 `dataset_train_all_현재시간.tsv`, `dataset_valid_all_현재시간.tsv`라는 이름으로 저장합니다. 저장 시 각 항목은 탭으로 구분됩니다.



# `colab_training_code.ipynb`
- KoBART를 이용하여 데이터셋을 학습 (colab)

# `main.ipynb`
- 파이썬 tkinter 라이브러리를 이용하여 GUI 설정
- 




---
### 데이터셋, 모델(학습 가중치, 토크나이저), exe실행파일 다운로드
- [dataset 다운로드](https://1drv.ms/u/s!AqYjAcj1n44riqRP0_B2RrlL1h9KyA?e=LCjHa1)

- [model 다운로드](https://1drv.ms/u/s!AqYjAcj1n44riqRQN0OJ3MjBrvAuZw?e=PiJxcI)

- [exe파일 다운로드](https://www.naver.com/)
