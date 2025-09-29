"""
RAG 파이프라인 모듈
PDF 문서 처리, 텍스트 분할, 임베딩 생성 및 벡터 DB 저장을 담당합니다.
"""

import fitz  # PyMuPDF
import io
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from milvus_client import MilvusClient


class RAGPipeline:
    """RAG 파이프라인 클래스"""
    
    def __init__(self, milvus_client: MilvusClient):
        """
        RAG 파이프라인 초기화
        
        Args:
            milvus_client: Milvus 클라이언트 인스턴스
        """
        self.milvus_client = milvus_client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        PDF 파일에서 텍스트 추출
        
        Args:
            pdf_content: PDF 파일 바이트 내용
            filename: PDF 파일명
            
        Returns:
            페이지별 텍스트와 메타데이터 리스트
        """
        try:
            # PDF 문서 열기
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            pages_data = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # 텍스트 추출
                text = page.get_text()
                
                if text.strip():  # 빈 페이지 제외
                    pages_data.append({
                        "text": text.strip(),
                        "page_number": page_num + 1,
                        "source_document": filename,
                        "total_pages": pdf_document.page_count
                    })
            
            pdf_document.close()
            print(f"PDF '{filename}'에서 {len(pages_data)}개 페이지의 텍스트를 추출했습니다.")
            return pages_data
            
        except Exception as e:
            raise RuntimeError(f"PDF 텍스트 추출 실패 ({filename}): {e}")
    
    def split_text(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할
        
        Args:
            pages_data: 페이지별 텍스트 데이터
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        try:
            chunks = []
            
            for page_data in pages_data:
                # 페이지 텍스트를 청크로 분할
                page_chunks = self.text_splitter.split_text(page_data["text"])
                
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    if chunk_text.strip():  # 빈 청크 제외
                        chunks.append({
                            "text": chunk_text.strip(),
                            "source_document": page_data["source_document"],
                            "page_number": page_data["page_number"],
                            "chunk_index": chunk_idx + 1,
                            "total_chunks": len(page_chunks)
                        })
            
            print(f"총 {len(chunks)}개의 텍스트 청크로 분할되었습니다.")
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"텍스트 분할 실패: {e}")
    
    def process_pdf_document(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        PDF 문서 전체 처리 파이프라인 실행
        
        Args:
            pdf_content: PDF 파일 바이트 내용
            filename: PDF 파일명
            
        Returns:
            처리 결과 정보 (문서명, 저장된 벡터 수 등)
        """
        try:
            print(f"PDF 문서 처리 시작: {filename}")
            
            # 1. PDF에서 텍스트 추출
            pages_data = self.extract_text_from_pdf(pdf_content, filename)
            
            if not pages_data:
                raise ValueError(f"PDF '{filename}'에서 추출할 수 있는 텍스트가 없습니다.")
            
            # 2. 텍스트 분할
            chunks = self.split_text(pages_data)
            
            if not chunks:
                raise ValueError(f"PDF '{filename}'에서 생성할 수 있는 텍스트 청크가 없습니다.")
            
            # 3. 벡터 DB에 저장
            inserted_count = self.milvus_client.insert_documents(chunks)
            
            result = {
                "filename": filename,
                "total_pages": pages_data[0]["total_pages"] if pages_data else 0,
                "extracted_pages": len(pages_data),
                "total_chunks": len(chunks),
                "inserted_vectors": inserted_count,
                "status": "success"
            }
            
            print(f"PDF 문서 처리 완료: {filename}")
            print(f"  - 추출된 페이지: {result['extracted_pages']}")
            print(f"  - 생성된 청크: {result['total_chunks']}")
            print(f"  - 저장된 벡터: {result['inserted_vectors']}")
            
            return result
            
        except Exception as e:
            error_result = {
                "filename": filename,
                "total_pages": 0,
                "extracted_pages": 0,
                "total_chunks": 0,
                "inserted_vectors": 0,
                "status": "error",
                "error_message": str(e)
            }
            
            print(f"PDF 문서 처리 실패: {filename} - {e}")
            return error_result
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        문서 검색 수행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 수
            
        Returns:
            유사한 문서 청크 리스트
        """
        try:
            print(f"문서 검색 수행: '{query}' (top_k={top_k})")
            
            # Milvus에서 유사도 검색
            similar_docs = self.milvus_client.search_similar(query, top_k)
            
            print(f"검색 완료: {len(similar_docs)}개 결과 반환")
            return similar_docs
            
        except Exception as e:
            print(f"문서 검색 실패: {e}")
            raise RuntimeError(f"문서 검색 실패: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """파이프라인 통계 정보 조회"""
        try:
            collection_info = self.milvus_client.get_collection_info()
            return {
                "pipeline_status": "active",
                "milvus_connection": collection_info["status"],
                "total_documents": collection_info["total_entities"],
                "text_splitter_config": {
                    "chunk_size": self.text_splitter._chunk_size,
                    "chunk_overlap": self.text_splitter._chunk_overlap
                }
            }
        except Exception as e:
            return {
                "pipeline_status": "error",
                "error_message": str(e)
            }
