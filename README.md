# VMware 운영 가이드 RAG 파이프라인

PDF 형식의 VMware 운영 가이드, 장애 처리 매뉴얼, 기술 문서를 벡터화하여 유사도 검색을 제공하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- **PDF 문서 처리**: PyMuPDF를 사용한 텍스트 추출
- **텍스트 분할**: LangChain의 RecursiveCharacterTextSplitter로 의미 있는 청크 생성
- **벡터 임베딩**: 한국어 특화 모델 `jhgan/ko-sroberta-multitask` 사용
- **벡터 저장**: Milvus 벡터 데이터베이스에 768차원 벡터 저장
- **유사도 검색**: 쿼리 기반 상위 5개 관련 문서 조각 반환
- **REST API**: FastAPI 기반 웹 API 제공

## 시스템 요구사항

- Python 3.8+
- Milvus 벡터 데이터베이스
- 8GB+ RAM (임베딩 모델 로딩용)

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Milvus 서버 실행

Docker를 사용한 Milvus 실행:

```bash
# Milvus Standalone 실행
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus:/var/lib/milvus \
  milvusdb/milvus:latest \
  milvus run standalone
```

### 3. 환경 변수 설정

`env.example` 파일을 `.env`로 복사하고 설정을 수정하세요:

```bash
cp env.example .env
```

### 4. 애플리케이션 실행

```bash
python main.py
```

또는 uvicorn으로 직접 실행:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 사용법

### 1. 문서 업로드

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@vmware_guide.pdf"
```

### 2. 문서 검색

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "VMware vSphere 가상화 설정 방법"}'
```

### 3. 시스템 상태 확인

```bash
curl -X GET "http://localhost:8000/health"
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | API 정보 |
| GET | `/health` | 시스템 헬스 체크 |
| POST | `/upload` | PDF 파일 업로드 및 처리 |
| POST | `/query` | 문서 유사도 검색 |
| GET | `/stats` | 시스템 통계 정보 |

## 프로젝트 구조

```
├── main.py              # FastAPI 애플리케이션
├── milvus_client.py     # Milvus 벡터 DB 클라이언트
├── rag_pipeline.py      # RAG 파이프라인 로직
├── requirements.txt     # Python 의존성
├── env.example         # 환경 변수 예시
└── README.md           # 프로젝트 문서
```

## 기술 스택

- **API 서버**: FastAPI
- **LLM 오케스트레이션**: LangChain
- **PDF 처리**: PyMuPDF
- **텍스트 분할**: LangChain RecursiveCharacterTextSplitter
- **임베딩 모델**: jhgan/ko-sroberta-multitask
- **벡터 DB**: Milvus
- **웹 프레임워크**: FastAPI + Uvicorn

## 라이선스

MIT License
