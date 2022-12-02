# 국방 AI 경진대회 코드 사용법

> 팀명 : 중요한 건 꺾는 마음

정시현, 이준형, 김태민, 임서현  
sh2298, jjuun, taemin6697, 임리둥절

# 핵심 파일 설명

- 학습 데이터 경로: `./data`
- 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 1개
  - `./results/train/final-submission`
- 학습 메인 코드: `./train.py`
- 테스트 메인 코드: `./predict.py`
- 테스트 이미지, 마스크 경로: `./results/pred/final-submission/mask `

## 코드 구조 설명

> 학습 진행하기 전 이미지 증강을 먼저 실행하여 학습 시간 단축

- **전처리**
  Get 1, 2 Labeled Image path

  - `python ./modules/get_class_path.py`

  Data Upsampling via ChangeFormer Augmentation Methods : `./preprocess/augmentation.py`

  - `python ./preprocess/augmentation.py`

  Data Upsampling via Cutmix using DFS Algorithm only for Label 1, 2 Mixing : `./preprocess/cutmix.py`

  - `python ./preprocess/cutmix.py`

  최종 학습 사용 데이터 (총 17559 장의 이미지 사용)

  - `./data/train`
  - `./data/up`
  - `./data/cut`

- **모델  
  nvidia/segformer-b4-finetuned-cityscapes-1024-1024**를 사용하여 학습 및 테스트함.  
  [Pretrained Model URL](https://huggingface.co/nvidia/segformer-b4-finetuned-cityscapes-1024-1024)

  ### # of Steps : 18000 steps

  Hyper-paramerter

  - batch:4
  - val_size: 0.3
  - learning_rate=0.00005
  - eval_accumulation_steps=10
  - epoch : 6

  최종 사용 모델 :
  nvidia/segformer-b4-finetuned-cityscapes-1024-1024 (Transfer Learning in 18000 steps lr 0.00005)

- **predict**
  `python predict.py`

- **최종 제출 파일 : ./results/pred/final-submission/mask**
- **학습된 가중치 파일 : ./results/train/final-submission** (Huggingface 특성상 폴더 전체를 불러와 모델에 연결시킴)

## 주요 설치 library

- torch==1.12.1
- torchvision==0.13.1
- transformers
- datasets
- pickle5
- pyyaml
- scikit-learn
- numpy
- pandas
- evaluate
- pillow
- opencv-python
- matplotlib

# How to use step by step

### 1. 처음부터 학습 및 추론하기

```
data 폴더 안에 train, test 이미지 넣기

pip install -r requirements.txt
python ./modules/get_class_path.py
python ./preprocess/augmentation.py
python ./preprocess/cutmix.py

./config/train.yaml 에서 train_folder 를 원하는 이름으로 수정
./config/predict.yaml 에서 train_folder 를 위와 같은 이름으로 수정

python train.py

학습이 전부 끝난 후
python predict.py
```

### 2. 최종 제출 가중치 파일로 추론하기

```
코드 수정없이
data 폴더 안에 train, test 이미지 넣기

pip install -r requirements.txt
python ./modules/get_class_path.py
python ./preprocess/augmentation.py
python ./preprocess/cutmix.py


python predict.py
```

## Architecture

```
army-ai
├─ config
│  ├─ predict.yaml
│  └─ train.yaml
| data
|  ├─ up
|  ├─ cut
|  ├─ train
|  ├─ pickle
|  └─ test
├─ models
│  └─ utils.py
├─ modules
│  ├─ EDA.py
│  ├─ datasets.py
│  ├─ earlystoppers.py
│  ├─ get_class_path.py
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ optimizers.py
│  ├─ recorders.py
│  ├─ scalers.py
│  ├─ schedulers.py
│  ├─ trainer.py
│  └─ utils.py
├─ preprocess
│  ├─ augmentation.py
│  ├─ class1.csv
│  ├─ class2.csv
│  └─ cutmix.py
├─ results
│  ├─ pred
│  │  └─ final-submission
│  │     ├─ mask
│  │     ├─ pred.log
│  │     ├─ predict_config.yml
│  │     └─ train_config.yml
│  └─ train
│     └─ final-submission
│        └─ checkpoint-18000
│           ├─ config.json
│           ├─ optimizer.pt
│           ├─ pytorch_model.bin
│           ├─ rng_state.pth
│           ├─ scheduler.pt
│           ├─ trainer_state.json
│           └─ training_args.bin
├─ Final-submit-mask.zip
├─ README.md
├─ predict.py
├─ requirements.txt
└─ train.py
```
