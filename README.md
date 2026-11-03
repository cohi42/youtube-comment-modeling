# YouTube Shorts Comment Preference Modeling

> 2025-2학기 **AI집중교육2** 팀 프로젝트

YouTube Shorts 댓글 생성 프로젝트를 정리한 연구형 포트폴리오입니다.

처음의 가설은 단순했습니다.

> 좋아요를 많이 받은 댓글을 학습하면, 좋아요를 받을 만한 댓글을 생성할 수 있지 않을까?

하지만 프로젝트를 진행하면서 이 가설은 곧바로 흔들렸습니다. 댓글의 좋아요 수는 댓글 자체의 품질만을 의미하지 않았습니다. 영상 조회수, 댓글 작성 시점, 댓글 노출 위치, 플랫폼 알고리즘, 채널 팬덤 효과가 모두 섞인 불완전한 신호였습니다.

따라서 이 프로젝트의 핵심은 단순한 댓글 생성 모델이 아니라, **좋아요라는 왜곡된 proxy signal을 학습 가능한 사용자 선호 데이터로 근사하는 과정**입니다.

## 핵심 질문

- "좋은 댓글"을 무엇으로 정의할 수 있는가?
- 댓글 좋아요 수는 정말 사용자 선호를 나타내는가?
- 조회수와 작성 시간의 영향을 어떻게 보정할 수 있는가?
- 영상과 무관한 댓글을 어떻게 걸러낼 수 있는가?
- 이렇게 정제한 데이터로 멀티모달 댓글 생성 모델을 학습하면 무엇이 달라지는가?

## 프로젝트 요약

이 프로젝트는 YouTube Shorts, 특히 음식 도메인 숏폼 영상을 대상으로 진행했습니다. 목표는 영상의 시각/청각 맥락을 이해하고, 사용자 반응을 이끌어낼 가능성이 높은 댓글을 생성하는 것이었습니다.

다만 최종적으로 더 중요하게 남은 기여는 모델 성능 자체보다, 다음과 같은 데이터 설계 과정이었습니다.

1. YouTube API와 수집 스크립트를 통해 영상 메타데이터와 댓글 데이터를 수집
2. 원본 댓글에서 스팸, 단순 반응, 영상과 무관한 댓글을 제거
3. 댓글 좋아요 수가 영상 조회수와 작성 시점에 의해 왜곡되는 문제를 분석
4. 시간 감쇄 모델과 조회수 기반 scale model로 좋아요 수를 보정
5. 멀티모달 LLM judge를 이용해 영상-댓글 관련성 점수를 추가
6. 정제된 댓글 후보로 Qwen3-Omni 기반 LoRA/SFT 댓글 생성 실험 수행
7. Few-shot, LoRA/SFT, modality 조합별 결과를 비교하고 한계를 분석

## 저장소 구조

```text
.
├── docs/
│   ├── problem_definition.md
│   ├── bias_correction.md
│   ├── evaluation_design.md
│   ├── experiments.md
│   ├── reflection.md
│   └── data_ethics.md
├── examples/
│   ├── sample_comments.anonymized.json
│   └── sample_dataset.schema.json
├── reference/
│   └── README.md
├── src/
│   ├── data_collection/
│   │   ├── youtube_shorts_scraper.py   # Selenium + YouTube Data API로 Shorts·댓글 수집
│   │   ├── video_downloader.py         # yt-dlp 영상 다운로드
│   │   └── youtube_schema.py           # 수집 JSON 로드·평탄화
│   ├── preprocessing/
│   │   ├── comment_cleaner.py          # 댓글 정제·필터링
│   │   └── media_extraction.py         # 영상 → 장면 프레임 + 16kHz 오디오
│   ├── bias_correction/
│   │   ├── time_decay.py               # 시간 감쇄 모델과 curve fitting
│   │   ├── view_scale.py               # 조회수 멱법칙 모델과 log-log 회귀
│   │   └── preference_score.py         # 이중 보정 점수 (final_score)
│   ├── relevance_scoring/
│   │   ├── prompts.py                  # 관련성 judge·생성 평가 프롬프트
│   │   └── qwen_judge.py               # Qwen3-Omni judge로 관련성 점수 부여
│   ├── dataset_building/
│   │   └── build_sft_dataset.py        # 후보 선정 → 최종 학습 댓글 선별
│   ├── training/
│   │   └── qwen_omni_lora.py           # Qwen3-Omni LoRA/SFT 학습
│   └── inference/
│       ├── describe_video.py           # 4-bit 모델로 영상 텍스트 설명 생성
│       └── generate_comment.py         # LoRA 모델로 댓글 생성
├── .env.example
├── .gitignore
├── requirements.txt
└── requirements-optional-model.txt
```

## 핵심 방법

### 1. 댓글 좋아요 수의 한계

댓글 좋아요 수는 직관적인 engagement 지표지만, 댓글 품질만을 반영하지 않습니다.

- 조회수가 높은 영상의 댓글은 더 많은 좋아요 기회를 가짐
- 영상 업로드 직후 작성된 댓글은 더 오래 노출됨
- 상단에 노출된 댓글은 양의 피드백 루프를 얻음
- 실제 댓글 노출량과 위치 변화는 공개 API로 관측할 수 없음

### 2. 편향 보정

이 프로젝트에서는 관측 가능한 변수로 두 가지 보정을 시도했습니다.

- **Time Decay Correction**: 영상 업로드 후 댓글 작성까지 걸린 시간에 따른 노출 이점 보정
- **View Scale Correction**: 영상 조회수 규모에 따른 기대 댓글 좋아요 수 보정

최종적으로 댓글의 절대 좋아요 수가 아니라, 조건 대비 얼마나 잘 반응을 얻었는지를 보는 proxy score를 만들었습니다.

```text
final_score = 좋아요 수 / (조회수 기반 기대 좋아요 × 작성 시점 기반 달성률)
```

조회수 5,000회 미만 영상과 노출 24시간 미만 댓글은 제외하고, 영상당 `final_score` 상위 20개를 후보로 선정했습니다.

### 3. 관련성 평가

좋아요가 높더라도 영상과 무관한 댓글은 학습 데이터로 적합하지 않습니다. 따라서 Qwen3-Omni를 judge로 사용해, 영상 프레임과 오디오를 함께 보고 후보 댓글 10개씩 관련성 점수(0-100)를 매겼습니다.

최종 학습 데이터는 **관련성 점수를 1순위, `final_score`를 2순위**로 정렬해 영상당 10개를 선별했습니다.

### 4. 모델 실험

정제한 데이터로 Qwen3-Omni 기반 멀티모달 댓글 생성 실험을 수행했습니다.

- Zero-shot / Few-shot baseline
- Text description 기반 입력
- Image 기반 입력
- Image + Audio 기반 입력
- LoRA/SFT 및 ORPO 시도

LoRA/SFT는 Qwen3-Omni-30B-A3B를 BF16으로 불러와 q/k/v/o projection에 LoRA(r=64, alpha=16)를 붙여 학습했습니다. 영상은 장면 전환 기준으로 추출한 프레임(최대 8장)과 16kHz 오디오로 입력했습니다.

실험 결과, 제한된 데이터와 하드웨어 조건에서는 Few-shot baseline이 더 강한 경우가 있었고, PEFT 성능은 기대만큼 나오지 않았습니다. 이 실패는 오히려 문제 정의, label 설계, evaluation 설계의 중요성을 보여주는 결과였습니다.

## 공개하지 않는 것

이 저장소에는 다음을 포함하지 않습니다.

- 원본 YouTube 영상 파일
- 원본 댓글 전체 데이터
- YouTube API 키, Hugging Face 토큰 (코드에서는 `[API Key]`로 마스킹)
- 수집 대상 채널 목록과 영상 ID
- 다운로드된 바이너리 파일
- 출력이 남아 있는 원본 notebook

대신 `examples/`에는 공개 가능한 익명화 샘플과 데이터 스키마만 포함했습니다.

`src/`의 코드는 Colab notebook과 로컬 스크립트로 진행한 원본 실험을 모듈 단위로 재정리한 것입니다. 수집, 학습, 추론 코드는 원본 흐름과 파라미터를 그대로 따르지만, API 키와 GPU, 비공개 영상 데이터가 필요해 이 저장소만으로는 실행되지 않습니다.

## 간단한 사용 예시

핵심 분석 의존성만 설치하려면 다음을 사용합니다.

```bash
pip install -r requirements.txt
```

Qwen-Omni LoRA/SFT reference 실험은 별도 GPU 환경이 필요하므로 의존성을 분리했습니다.

```bash
pip install -r requirements-optional-model.txt
```

익명화 샘플 댓글로 `final_score` 계산과 최종 댓글 선별 과정을 재현해볼 수 있습니다. 이 단계는 GPU 없이 실행됩니다.

```bash
python -m src.dataset_building.build_sft_dataset \
  --input examples/sample_comments.anonymized.json \
  --output outputs/sample_sft_dataset.json
```

## 회고

이 프로젝트는 제 첫 실질적인 AI/NLP 프로젝트에 가까웠습니다. 처음에는 "좋아요를 많이 받은 댓글을 학습하면 좋은 댓글을 만들 수 있지 않을까?"라는 단순한 생각으로 시작했지만, 실제로는 "좋은 댓글"의 정의부터 명확하지 않았습니다.

특히 NLP 문제에서 evaluation이 얼마나 어려운지, 그리고 모델 학습보다 문제 정의와 label 설계가 더 근본적인 병목이 될 수 있다는 점을 배웠습니다.

모델이 실패한 것처럼 보이는 순간에도, 그 실패가 단순한 구현 실패인지, 데이터 정의 실패인지, 평가 설계 실패인지 구분하는 일이 중요했습니다. 이 저장소는 그 과정을 정리한 결과입니다.

## 관련 문서

- [문제 정의](docs/problem_definition.md)
- [편향 보정](docs/bias_correction.md)
- [평가 설계](docs/evaluation_design.md)
- [실험 결과와 한계](docs/experiments.md)
- [프로젝트 회고](docs/reflection.md)
- [데이터 윤리](docs/data_ethics.md)
