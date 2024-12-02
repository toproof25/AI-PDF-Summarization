## AI-PDF-Summarization

### 목차
- [create_dataset_to_tsv.ipynb](#create_dataset_to_tsvipynb)
- [colab_training_code.ipynb](#colab_training_codeipynb)
- [main.ipynb](#mainipynb)
- [데이터셋, 모델(학습 가중치, 토크나이저), exe실행파일 다운로드](#데이터셋, 모델(학습 가중치, 토크나이저), exe실행파일 다운로드)



### dataset
- [AI Hub 요약문 및 레포트 생성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=582)

### KoBART-summarization
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)


<br/>
<br/>
<br/>

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

<br/>
<br/>
<br/>

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

#### 주요 옵션

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


<br/>
<br/>
<br/>



## `main.ipynb`
- 파이썬 tkinter 라이브러리를 이용하여 GUI 설정

### 1. 학습 모델 불러오기
```python
import torch # 2.0.1
from transformers import PreTrainedTokenizerFast # 4.32.1
from transformers.models.bart import BartForConditionalGeneration

from tkinter import scrolledtext, filedialog
import sys
import tkinter as tk 
import fitz  # PDF 읽기
import threading  # 스레딩 모듈 
import re # 정규화

# KoBART 모델 및 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('./model/tokenizer')
model = BartForConditionalGeneration.from_pretrained('./model/model')
```


### 2. 텍스트 입력 시 요약문을 반환하는 함수
```python
def summarize_text(input_text, max_length=150, min_length=40, num_beams=5, length_penalty=1.2, early_stopping=True):
    # 입력 텍스트를 토큰화하여 인코딩
    input_ids = tokenizer.encode(       # 입력 텍스트를 단어 ID로 변환
        input_text,                     # 입력 텍스트
        return_tensors="pt",            # pytorch로 변환 -> 모델에 입력하기 위해
        max_length=1024,                # 최대 길이 1024
        truncation=True                 # 최대 길이가 초과되면 텍스트 자르기
    )

    # 모델을 사용하여 요약 생성
    summary_ids = model.generate(       # 문장 생성 함수 
        input_ids,                      # 인코드한 변수
        max_length=max_length,          # 요약문 최대 길이
        min_length=min_length,          # 요약문 최소 길이
        length_penalty=length_penalty,  # 요약문의 길이 설정 -> 값이 높을수록 길이가 짧아짐
        num_beams=num_beams,            # 단어 후보의 수 -> 단어 후보 4개에서 하나를 선택 -> 클수록 정확도 향상 
        early_stopping=early_stopping   # 더 이상 생성되는 단어가 없으면 일찍 종료할 수 있는 기능
    )

    # 요약 결과 디코딩
    summary_text = tokenizer.decode(    # 단어 ID를 다시 텍스트로 변환
        summary_ids[0],                 # 요약문에 해당하는 인덱스
        skip_special_tokens=True        # 의미 있는 텍스트만 변환함 
    )        
    
    return summary_text                 # 최종 요약 문장
```

### 3. GUI 설정
- ##### 윈도우 사이즈 800 x 900
- ##### 구성요소
  - **원문 입력**: 요약 전 원문을 입력 및 표시되는 창
  - **예측 요약문**: 최종적으로 요약된 텍스트가 표시되는 창
  - **최대, 최소 길이 입력, num_beams, length_penalty, early_stopping**: 생성 파라미터 제어창
  - **PDF 불러오기 버튼**: 파일 탐색기를 이용하여 PDF를 불러오는 버튼
  - **테스트 원문 생성**: 테스트 원문을 입력창에 입력하는 버튼
  - **요약문 생성**: 원문 입력에 있는 텍스트를 요약하는 버튼
  - **설정 초기화**: 각 파라미터 값 초기화

### 4. 주요 함수 설명
#### `on_load_pdf_button_click()`
- **기능**: PDF 파일을 불러오기 위해 사용됩니다.
- **동작**: 파일 탐색기를 통해 PDF 파일을 선택하고, 해당 파일에서 텍스트를 추출하여 원문 입력창에 표시합니다.

#### `on_summarize_button_click()`
- **기능**: 입력된 원문을 요약하는 버튼 클릭 시 실행됩니다.
- **동작**:
  - 입력된 텍스트에서 한글만 추출하여, 불필요한 영어 및 특수문자를 제거합니다.
  - `summarize_text()` 함수에 텍스트를 전달하여 요약을 생성하고, 결과를 예측 요약문 창에 표시합니다.

#### 기타 슬라이더 및 버튼
- **슬라이더**: 최대 길이, 최소 길이, 빔 수, 길이 패널티 등을 설정할 수 있는 슬라이더가 있습니다.
- **Early Stopping 체크박스**: Early Stopping을 활성화할 수 있는 옵션을 제공합니다.
- **설정 초기화 버튼**: 슬라이더와 체크박스의 설정을 초기화할 수 있는 버튼을 제공합니다.
- **테스트 원문 생성 버튼**: 미리 정의된 테스트 원문을 원문 입력창에 불러오는 버튼을 제공합니다.

### 코드 흐름
1. **GUI 구성**: `tkinter`를 이용하여 윈도우, 레이블, 텍스트 박스, 버튼 등을 배치합니다.
2. **PDF 불러오기**: `filedialog.askopenfilename()`을 사용하여 PDF 파일을 선택하고 텍스트를 추출하여 입력창에 표시합니다.
3. **요약 생성**: 사용자가 설정한 파라미터(최대 길이, 최소 길이, 빔 수, 길이 패널티 등)에 맞게 텍스트를 요약합니다.



<br/>
<br/>
<br/>


---
### 데이터셋, 모델(학습 가중치, 토크나이저), exe실행파일 다운로드
- [dataset (tsv) 다운로드](https://1drv.ms/u/s!AqYjAcj1n44riqRP0_B2RrlL1h9KyA?e=LCjHa1)
- [model 다운로드](https://1drv.ms/u/s!AqYjAcj1n44riqRQN0OJ3MjBrvAuZw?e=PiJxcI)
- [exe파일 다운로드](https://1drv.ms/u/s!AqYjAcj1n44ri4ZI3mKbJ1CxBnPoZg?e=7KXQ1x)
