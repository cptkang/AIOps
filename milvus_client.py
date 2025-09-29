"""
Milvus 벡터 데이터베이스 클라이언트 모듈
Milvus 서버 연결, 컬렉션 관리, 벡터 삽입 및 검색 기능을 제공합니다.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, Index
)
from sentence_transformers import SentenceTransformer
import numpy as np


class MilvusClient:
    """Milvus 벡터 데이터베이스 클라이언트"""
    
    def __init__(self, host: str = None, port: str = None, collection_name: str = "vmware_docs"):
        """
        Milvus 클라이언트 초기화
        
        Args:
            host: Milvus 서버 호스트 (기본값: 환경변수 MILVUS_HOST)
            port: Milvus 서버 포트 (기본값: 환경변수 MILVUS_PORT)
            collection_name: 컬렉션 이름
        """
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.collection_name = collection_name
        self.embedding_model = None
        self.collection = None
        
        # Milvus 연결
        self._connect()
        
        # 임베딩 모델 로드
        self._load_embedding_model()
        
        # 컬렉션 설정
        self._setup_collection()
    
    def _connect(self):
        """Milvus 서버에 연결"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print(f"Milvus 서버에 연결되었습니다: {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"Milvus 서버 연결 실패: {e}")
    
    def _load_embedding_model(self):
        """한국어 임베딩 모델 로드"""
        try:
            self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            print("임베딩 모델이 로드되었습니다: jhgan/ko-sroberta-multitask")
        except Exception as e:
            raise RuntimeError(f"임베딩 모델 로드 실패: {e}")
    
    def _setup_collection(self):
        """컬렉션 스키마 설정 및 생성"""
        # 컬렉션이 이미 존재하는지 확인
        if utility.has_collection(self.collection_name):
            print(f"컬렉션 '{self.collection_name}'이 이미 존재합니다.")
            self.collection = Collection(self.collection_name)
        else:
            print(f"컬렉션 '{self.collection_name}'을 생성합니다.")
            self._create_collection()
        
        # 컬렉션 로드
        self.collection.load()
    
    def _create_collection(self):
        """새 컬렉션 생성"""
        # 필드 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        
        # 컬렉션 스키마 생성
        schema = CollectionSchema(
            fields=fields,
            description="VMware 문서 벡터 저장소",
            enable_dynamic_field=True
        )
        
        # 컬렉션 생성
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # 인덱스 생성 (IVF_FLAT)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print(f"컬렉션 '{self.collection_name}'이 성공적으로 생성되었습니다.")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        텍스트를 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 (768차원)
        """
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"임베딩 생성 실패: {e}")
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        문서들을 벡터 DB에 삽입
        
        Args:
            documents: 삽입할 문서 리스트 (text, source_document 포함)
            
        Returns:
            삽입된 문서 수
        """
        if not documents:
            return 0
        
        try:
            # 데이터 준비
            ids = []
            texts = []
            sources = []
            embeddings = []
            
            for doc in documents:
                # 고유 ID 생성
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # 텍스트와 소스 저장
                texts.append(doc["text"])
                sources.append(doc["source_document"])
                
                # 임베딩 생성
                embedding = self.generate_embedding(doc["text"])
                embeddings.append(embedding)
            
            # Milvus에 삽입
            data = [ids, texts, sources, embeddings]
            self.collection.insert(data)
            self.collection.flush()
            
            print(f"{len(documents)}개 문서가 성공적으로 삽입되었습니다.")
            return len(documents)
            
        except Exception as e:
            raise RuntimeError(f"문서 삽입 실패: {e}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        유사도 검색 수행
        
        Args:
            query: 검색 쿼리 텍스트
            top_k: 반환할 상위 결과 수
            
        Returns:
            유사한 문서 리스트 (text, source_document, distance 포함)
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.generate_embedding(query)
            
            # 검색 파라미터 설정
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 유사도 검색 수행
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source_document"]
            )
            
            # 결과 포맷팅
            similar_docs = []
            for hit in results[0]:
                similar_docs.append({
                    "text": hit.entity.get("text"),
                    "source_document": hit.entity.get("source_document"),
                    "distance": float(hit.distance),
                    "id": hit.id
                })
            
            return similar_docs
            
        except Exception as e:
            raise RuntimeError(f"유사도 검색 실패: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            stats = self.collection.get_stats()
            return {
                "collection_name": self.collection_name,
                "total_entities": stats.get("row_count", 0),
                "status": "connected"
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "total_entities": 0,
                "status": f"error: {e}"
            }
    
    def close(self):
        """연결 종료"""
        try:
            connections.disconnect("default")
            print("Milvus 연결이 종료되었습니다.")
        except Exception as e:
            print(f"연결 종료 중 오류: {e}")
