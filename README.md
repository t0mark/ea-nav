# 데이터 다운로드

## 1. 업로드 방법

```bash
hf upload data/ . --repo-type dataset --exclude "isaac-cache/**"
```

## 2. 다운로드

```bash
hf download t0mark/ea-nav --repo-type dataset --local-dir data/
```
