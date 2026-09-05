# 데이터 다운로드

`data/` 에셋은 용량 문제로 Hugging Face에 별도로 업로드되어 있습니다. 아래 방법으로 받으세요.

## 1. huggingface_hub 설치

```bash
pip install -U "huggingface_hub[cli]"
```

## 2. 다운로드

```bash
hf download t0mark/ea-nav --repo-type dataset --local-dir data/
```
