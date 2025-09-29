"""
VMware 운영 가이드 RAG 파이프라인 FastAPI 애플리케이션
PDF 문서 업로드, 처리, 벡터 저장 및 유사도 검색 API를 제공합니다.
"""

import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from milvus_client import MilvusClient
from rag_pipeline import RAGPipeline

# 환경 변수 로드
load_dotenv()

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="VMware 운영 가이드 RAG 파이프라인",
    description="PDF 문서를 벡터화하여 유사도 검색을 제공하는 RAG 시스템",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
milvus_client = None
rag_pipeline = None


def get_milvus_client() -> MilvusClient:
    """Milvus 클라이언트 의존성 주입"""
    global milvus_client
    if milvus_client is None:
        try:
            milvus_client = MilvusClient()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Milvus 연결 실패: {e}")
    return milvus_client


def get_rag_pipeline() -> RAGPipeline:
    """RAG 파이프라인 의존성 주입"""
    global rag_pipeline
    if rag_pipeline is None:
        try:
            client = get_milvus_client()
            rag_pipeline = RAGPipeline(client)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG 파이프라인 초기화 실패: {e}")
    return rag_pipeline


# Pydantic 모델 정의
class QueryRequest(BaseModel):
    """검색 쿼리 요청 모델"""
    query: str
    
    class Config:
        schema_extra = {
            "example": {
                "query": "VMware vSphere 가상화 설정 방법"
            }
        }


class QueryResponse(BaseModel):
    """검색 결과 응답 모델"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int


class UploadResponse(BaseModel):
    """업로드 결과 응답 모델"""
    filename: str
    total_pages: int
    extracted_pages: int
    total_chunks: int
    inserted_vectors: int
    status: str
    error_message: str = None


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    pipeline_info: Dict[str, Any]


# API 엔드포인트
@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "VMware 운영 가이드 RAG 파이프라인 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """시스템 헬스 체크"""
    try:
        pipeline_stats = pipeline.get_pipeline_stats()
        return HealthResponse(
            status="healthy" if pipeline_stats["pipeline_status"] == "active" else "unhealthy",
            pipeline_info=pipeline_stats
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            pipeline_info={"error": str(e)}
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    PDF 파일 업로드 및 처리
    
    - **file**: 업로드할 PDF 파일
    - **return**: 처리 결과 (문서명, 페이지 수, 청크 수, 저장된 벡터 수)
    """
    # 파일 타입 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )
    
    try:
        # 파일 내용 읽기
        pdf_content = await file.read()
        
        if len(pdf_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="빈 파일입니다."
            )
        
        # PDF 처리 파이프라인 실행
        result = pipeline.process_pdf_document(pdf_content, file.filename)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"PDF 처리 실패: {result.get('error_message', '알 수 없는 오류')}"
            )
        
        return UploadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 업로드 처리 중 오류 발생: {e}"
        )


@app.post("/query", response_model=QueryResponse)
async def search_documents(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    문서 유사도 검색
    
    - **query**: 검색할 텍스트 쿼리
    - **return**: 유사한 문서 청크 리스트 (상위 5개)
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="검색 쿼리가 비어있습니다."
        )
    
    try:
        # 문서 검색 수행
        results = pipeline.search_documents(request.query, top_k=5)
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문서 검색 중 오류 발생: {e}"
        )


@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """시스템 통계 정보 조회"""
    try:
        stats = pipeline.get_pipeline_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"통계 정보 조회 중 오류 발생: {e}"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 리소스 정리"""
    global milvus_client
    if milvus_client:
        milvus_client.close()
        print("애플리케이션이 종료되었습니다.")


if __name__ == "__main__":
    import uvicorn
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
