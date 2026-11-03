# Reference Notes

원본 실험은 Colab/Jupyter notebook 중심으로 진행되었습니다. 이 정리본에서는 notebook을 직접 공개 산출물로 두지 않습니다.

이유는 다음과 같습니다.

- notebook 출력에 실행 순서, 임시 경로, 에러 로그, 토큰 흔적이 남을 수 있음
- GitHub diff가 JSON 단위로 지저분하게 보임
- 포트폴리오에서 핵심 의사결정과 함수 구조를 읽기 어려움

필요하다면 이 폴더에는 다음 조건을 만족하는 reference notebook만 둘 수 있습니다.

1. 모든 output 제거
2. API key, token, 개인 경로 제거
3. 원본 댓글/영상 데이터 미포함
4. README 또는 `docs/` 문서와 역할이 중복되지 않도록 보조 자료로만 유지

현재 공개 정리본의 중심은 `src/`의 Python 모듈과 `docs/`의 연구 설명입니다.
