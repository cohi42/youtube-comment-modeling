# 실험 정리

## 모델 목표

최종 모델 실험의 목표는 다음과 같았다.

> 영상의 시각/청각 맥락을 이해하고, 사용자 선호를 이끌어낼 가능성이 높은 댓글을 생성한다.

모델은 Qwen3-Omni 계열 AVLM을 중심으로 실험했다. 영상 프레임과 오디오를 함께 입력하고, 정제된 고품질 댓글을 target으로 삼아 LoRA/SFT를 수행하는 방향이었다.

## 실험 구성

실험은 크게 다음 축으로 나눌 수 있다.

- Zero-shot 생성
- Few-shot 생성
- Text description 기반 생성
- Image 기반 생성
- Image + Audio 기반 생성
- LoRA/SFT fine-tuning
- ORPO 방식의 선호 학습 시도

## 평가 기준

발표자료에서는 Multimodal LLM-as-a-judge 방식으로 생성 댓글을 평가했다. 평가 모델은 댓글을 단순히 정답 문장과 비교하지 않고, 다음 세 축으로 나누어 0-100점 범위에서 판단하도록 설계했다.

| 평가 축 | 질문 | 감점되는 예 |
| --- | --- | --- |
| Contextual Grounding | 영상의 고유한 시각/청각 정보를 정확히 반영하는가? | 어느 영상에나 붙일 수 있는 범용 멘트, 영상에 없는 내용 |
| Semantic Resonance | 음식 도메인 시청자의 심리와 공감 포인트를 건드리는가? | 단순 captioning, 식욕/대리만족/상황 포인트 부재 |
| Human-like Tone | 실제 한국 YouTube 댓글처럼 자연스러운가? | 번역투, 지나치게 예의 바른 설명문, AI 요약문 |

최종 점수는 세 항목 평균으로 계산했다.

```text
FinalScore = mean(ContextualGrounding, SemanticResonance, HumanLikeTone)
```

## 데이터 구성

학습 데이터는 raw like count를 그대로 사용하지 않고 다음 과정을 거쳐 후보를 선별했다.

1. 댓글 전처리
2. 영상-댓글 관련성 평가
3. 시간 감쇄 보정
4. 조회수 scale 보정
5. preference proxy score 기준 상위 후보 선별

## 주요 관찰

### Modality별 결과

발표자료의 최종 결과 표를 정리하면 다음과 같다.

| 입력 설정 | Zero-shot | Few-shot | PEFT (LoRA) |
| --- | ---: | ---: | ---: |
| Text description | 78.47 | 82.20 | 64.60 |
| Image | 70.40 | 71.53 | 65.71 |
| Image + Audio | 69.93 | 78.56 | 66.24 |

몇 가지 관찰이 가능하다.

- Few-shot은 모든 입력 설정에서 PEFT보다 높은 점수를 보였다.
- Text description 기반 Few-shot이 82.20점으로 가장 높았다.
- Image + Audio는 Zero-shot에서 69.93점이었지만 Few-shot에서 78.56점으로 상승했다.
- Image-only Few-shot 71.53점과 비교하면, Audio를 추가한 설정은 78.56점으로 7.03점 높았다.

### 선호 학습 추가 실험

ORPO도 추가로 시도했지만, Image + Audio 설정에서는 SFT보다 낮은 점수를 보였다.

| 설정 | 점수 |
| --- | ---: |
| SFT | 66.24 |
| ORPO | 62.76 |

이 결과는 chosen/rejected 데이터를 구성하더라도, preference label 자체가 충분히 안정적이지 않으면 선호 학습이 자동으로 개선을 보장하지 않는다는 점을 보여준다.

### Few-shot baseline의 강함

제한된 데이터와 컴퓨팅 자원에서는 Few-shot baseline이 fine-tuning보다 더 좋은 결과를 보이는 경우가 있었다.

이는 base model이 이미 강한 언어/멀티모달 능력을 갖고 있으며, 작은 규모의 정제 데이터로는 그 능력을 안정적으로 개선하기 어렵다는 점을 보여준다.

### PEFT 성능의 제한

LoRA/SFT 실험은 수행했지만 성능 개선은 제한적이었다.

가능한 원인은 다음과 같다.

- Colab A100 환경의 VRAM 한계
- 입력 가능한 frame 수 제한
- 학습 데이터 규모 부족
- 댓글의 "좋음"을 나타내는 label 자체의 불안정성
- generation quality를 평가하는 기준의 어려움

### Text description의 의외의 강점

영상 전체를 충분히 넣지 못하는 상황에서는, 제한된 frame보다 사람이 작성하거나 모델이 생성한 text description이 더 많은 정보를 담을 수 있었다.

이는 멀티모달 입력이 항상 우월하다는 뜻이 아니라, 실제 하드웨어 제약 아래에서는 입력 표현 방식의 정보량과 안정성이 더 중요할 수 있음을 보여준다.

### Audio의 중요성

음식 Shorts에서는 조리 소리, ASMR, 말투, 분위기 같은 청각 정보가 댓글의 공감 포인트와 연결되는 경우가 있었다.

따라서 image-only 입력보다 audio를 함께 고려하는 설정이 더 설득력 있는 방향이었다. 실제 결과에서도 Few-shot 기준 `Image`는 71.53점, `Image + Audio`는 78.56점으로 더 높았다.

### 실제 생성 예시

Qwen3-Omni inference notebook에는 다음과 같은 생성 예시가 남아 있었다.

```text
입력 영상: "그렇게 맛있다고 극찬들로만 가득한 버거 드디어 먹어봤습니다"
생성 댓글: "이거 먹고 싶어도 맛집이라서 못먹겠네..."
```

이 출력은 댓글 형식 자체는 자연스럽지만, 왜 해당 영상만의 고유한 시각/청각 맥락을 반영하는지 충분히 설명하지 못한다. 이 때문에 단순 생성 가능 여부보다 contextual grounding 평가가 중요해졌다.

## 실패의 의미

이 프로젝트는 "fine-tuning을 했더니 성능이 확실히 좋아졌다"는 식의 깔끔한 성공 사례는 아니었다.

하지만 다음을 확인했다.

- 모델 성능보다 데이터 정의가 먼저 흔들릴 수 있다.
- 좋아요 수 같은 engagement metric은 학습 label로 쓰기 전에 반드시 해석이 필요하다.
- 평가 기준이 불명확하면 모델 비교도 불명확해진다.
- 제한된 자원에서는 fine-tuning보다 prompt/evaluation/data filtering의 영향이 더 클 수 있다.

이 실패는 단순한 구현 실패가 아니라, 답이 명확하지 않은 NLP 문제에서 무엇이 병목이 되는지 확인한 경험이었다.
