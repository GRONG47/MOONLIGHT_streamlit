# UCI SECOM ML 분석 대시보드

이 브랜치는 `uci-secom.csv` 기준으로 ML1, ML2, ML3 모델을 Streamlit에서 실행합니다.

## 데이터 재확인

- 행/열: 1567행 x 592열
- Pass: 1463개
- Fail: 104개
- 불량 비율: 약 6.64%
- CSV SHA256: `E8B8187CE276DB148707B5103463AA0F35CB8E521DA2BC697C0FD45A6B06AD0F`

앱은 CSV가 없으면 같은 해시의 공개 원본을 내려받고, 해시가 다르면 실행을 멈춥니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
