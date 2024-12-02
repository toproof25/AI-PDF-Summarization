# AI-PDF-Summarization


### dataset
- [AI Hub 요약문 및 레포트 생성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=582)

### KoBART-summarization
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)




## `create_dataset_to_tsv.ipynb`
- 데이터셋을 tsv로 추출하는 코드 (jupyter notebook)

### JSON 데이터 처리 및 TSV 파일 생성

이 프로젝트는 여러 폴더에 저장된 JSON 데이터를 읽고, 학습용 및 시험용 데이터를 처리하여 TSV 파일로 변환하는 파이썬 스크립트입니다.

1. **폴더 목록 지정**: 다양한 종류의 데이터를 포함하는 여러 폴더를 설정합니다.
2. **JSON 데이터 읽기**: 각 폴더에 있는 JSON 파일을 읽고, 필요한 데이터를 추출하여 리스트에 저장합니다.
3. **데이터 프레임 생성**: 추출한 데이터를 `pandas` DataFrame으로 변환합니다.
4. **TSV 파일로 저장**: 처리된 데이터를 TSV 형식으로 저장하여 외부 시스템에서 사용할 수 있도록 합니다.

### 사용 방법

#### 1. `folder_list`에 있는 폴더들에서 JSON 파일을 읽어들입니다.
각 폴더는 아래와 같은 형태로 구성되어 있으며, 각 JSON 파일은 두 가지 주요 항목을 포함합니다:
- `Meta(Refine)` → `passage`: 뉴스 본문
- `Annotation` → `summary1`: 뉴스 요약

#### 2. 데이터를 TSV 형식으로 변환합니다.
학습 데이터와 시험 데이터를 각각 `dataset_train_all_현재시간.tsv`, `dataset_valid_all_현재시간.tsv`라는 이름으로 저장합니다. 저장 시 각 항목은 탭으로 구분됩니다.



## `colab_training_code.ipynb`
- KoBART를 이용하여 데이터셋을 학습 (colab)

### 라이브러리 설치

- **pandas** : 데이터프레임 및 데이터 조작을 위한 라이브러리  

- **torch (PyTorch)** : 딥러닝 모델 구현 및 학습을 위한 라이브러리    

- **transformers** : 트랜스포머 기반 모델(BERT, GPT 등)을 사용하기 위한 라이브러리  

- **tokenizers** : 빠르고 효율적인 토크나이저 사용을 위한 라이브러리  

- **lightning (PyTorch Lightning)** : 간결한 모델 훈련 관리 및 학습 프레임워크  

- **streamlit** : 웹 애플리케이션을 빠르게 구축할 수 있는 라이브러리 (시각화용)  

- **wandb (Weights & Biases)** : 모델 학습 과정 모니터링 및 실험 관리 도구

- **loguru** : 고급 로깅 기능 제공을 위한 라이브러리  

- **rouge_score** : ROUGE 점수 계산을 위한 라이브러리 (요약 성능 평가)  
```python
# 패키지 설치
!pip install pandas
!pip install torch==2.0.1
!pip install transformers==4.32.1
!pip install tokenizers==0.13.3
!pip install lightning==2.0.8
!pip install streamlit==1.26.0
!pip install wandb==0.15.9
!pip install loguru
!pip install rouge_score
```

### 1. KoBERT 학습 시작

####주요 옵션

- gradient_clip_val 1.0 :
기울기 클리핑 값

- max_epochs 100 :
최대 에폭 수

- checkpoint checkpoint :
학습 중 모델 가중치와 상태를 저장할 디렉터리를 지정

- accelerator gpu :
GPU를 사용

- num_gpus 1 :
사용할 GPU 수

- batch_size 16 :
배치 크기 16

- num_workers 4 :
데이터를 로드하는 병렬 작업의 수를 4로 설정

```python
!python train.py --gradient_clip_val 1.0 \
                --max_epochs 100 \
                --checkpoint checkpoint \
                --accelerator gpu \
                --num_gpus 1 \
                --batch_size 16 \
                --num_workers 4
```

### 3. 학습된 모델 바이너리로 만들기 
```python
# 학습된 모델을 바이너리로 만듦
!python get_model_binary.py --model_binary '/content/drive/MyDrive/KoBART-summarization-main/KoBART-summarization-main/checkpoint/summarization_final/epoch=99-
```

### 4. 학습 모델 불러오기
```python
import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

# 모델 바이너리 파일 경로
model_binary_path = '/content/drive/MyDrive/KoBART-summarization-main/KoBART-summarization-main/kobart_summary'

# KoBART 모델 및 토크나이저 로드
model = BartForConditionalGeneration.from_pretrained(model_binary_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
```



## `main.ipynb`
- 파이썬 tkinter 라이브러리를 이용하여 GUI 설정
- 




---
### 데이터셋, 모델(학습 가중치, 토크나이저), exe실행파일 다운로드
- [dataset 다운로드](https://1drv.ms/u/s!AqYjAcj1n44riqRP0_B2RrlL1h9KyA?e=LCjHa1)

- [model 다운로드](https://1drv.ms/u/s!AqYjAcj1n44riqRQN0OJ3MjBrvAuZw?e=PiJxcI)

- [exe파일 다운로드](https://www.naver.com/)
