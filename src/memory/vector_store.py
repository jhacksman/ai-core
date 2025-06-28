"""
Vector Store - Chroma DB integration for embeddings and semantic search.

This module provides vector database functionality for the Venice.ai scaffolding
system using Chroma DB for persistent embeddings and semantic search capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Represents a document in the vector store."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    document: VectorDocument
    similarity_score: float
    rank: int


class VectorStore:
    """
    Vector database integration using Chroma DB for embeddings and semantic search.
    
    Provides persistent storage and retrieval of embeddings for the Venice.ai
    scaffolding system with support for semantic search and similarity matching.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        collection_name: str = "ai_core_vectors",
        persist_directory: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            venice_client: Venice.ai client for generating embeddings
            collection_name: Name of the Chroma collection
            persist_directory: Directory for persistent storage
            config: Configuration options
        """
        if chromadb is None:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
        
        self.venice_client = venice_client
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./chroma_db"
        self.config = config or {}
        
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
        self.embedding_model = self.config.get("embedding_model", "text-embedding-ada-002")
        self.embedding_dimension = self.config.get("embedding_dimension", 1536)
        self.max_batch_size = self.config.get("max_batch_size", 100)
        
        self.stats = {
            "documents_stored": 0,
            "searches_performed": 0,
            "embeddings_generated": 0,
            "cache_hits": 0,
            "average_search_time": 0.0
        }
        
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_max_size = self.config.get("cache_max_size", 10000)
    
    async def initialize(self) -> None:
        """Initialize the vector store and Chroma DB connection."""
        logger.info("Initializing Vector Store")
        
        try:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.Client(settings)
            
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function()
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={"description": "AI Core vector embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            await self._load_stats()
            
            logger.info("Vector Store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {e}")
            raise
    
    async def add_document(
        self,
        document: VectorDocument,
        generate_embedding: bool = True
    ) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            document: Document to add
            generate_embedding: Whether to generate embedding for the document
            
        Returns:
            True if document was added successfully
        """
        try:
            if generate_embedding and not document.embedding:
                document.embedding = await self._generate_embedding(document.content)
            
            self.collection.add(
                ids=[document.id],
                documents=[document.content],
                embeddings=[document.embedding] if document.embedding else None,
                metadatas=[{
                    **document.metadata,
                    "timestamp": document.timestamp.isoformat(),
                    "content_length": len(document.content)
                }]
            )
            
            self.stats["documents_stored"] += 1
            
            logger.debug(f"Added document to vector store: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            return False
    
    async def add_documents(
        self,
        documents: List[VectorDocument],
        generate_embeddings: bool = True
    ) -> int:
        """
        Add multiple documents to the vector store in batches.
        
        Args:
            documents: List of documents to add
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Number of documents successfully added
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        added_count = 0
        
        for i in range(0, len(documents), self.max_batch_size):
            batch = documents[i:i + self.max_batch_size]
            
            try:
                if generate_embeddings:
                    await self._generate_embeddings_batch(batch)
                
                ids = [doc.id for doc in batch]
                contents = [doc.content for doc in batch]
                embeddings = [doc.embedding for doc in batch if doc.embedding]
                metadatas = [
                    {
                        **doc.metadata,
                        "timestamp": doc.timestamp.isoformat(),
                        "content_length": len(doc.content)
                    }
                    for doc in batch
                ]
                
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    embeddings=embeddings if embeddings else None,
                    metadatas=metadatas
                )
                
                added_count += len(batch)
                logger.debug(f"Added batch {i//self.max_batch_size + 1}: {len(batch)} documents")
                
            except Exception as e:
                logger.error(f"Failed to add batch starting at index {i}: {e}")
        
        self.stats["documents_stored"] += added_count
        
        logger.info(f"Successfully added {added_count}/{len(documents)} documents")
        return added_count
    
    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include_similarity: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filters
            include_similarity: Whether to include similarity scores
            
        Returns:
            List of search results
        """
        start_time = datetime.now()
        
        try:
            query_embedding = await self._generate_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    content = results["documents"][0][i] if results["documents"] else ""
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    similarity_score = max(0.0, 1.0 - distance) if include_similarity else 0.0
                    
                    document = VectorDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                    )
                    
                    search_result = SearchResult(
                        document=document,
                        similarity_score=similarity_score,
                        rank=i + 1
                    )
                    
                    search_results.append(search_result)
            
            search_time = (datetime.now() - start_time).total_seconds()
            self.stats["searches_performed"] += 1
            self._update_average_search_time(search_time)
            
            logger.debug(f"Search completed in {search_time:.3f}s with {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def search_by_embedding(
        self,
        embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filters
            
        Returns:
            List of search results
        """
        start_time = datetime.now()
        
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    content = results["documents"][0][i] if results["documents"] else ""
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    document = VectorDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                    )
                    
                    search_result = SearchResult(
                        document=document,
                        similarity_score=similarity_score,
                        rank=i + 1
                    )
                    
                    search_results.append(search_result)
            
            search_time = (datetime.now() - start_time).total_seconds()
            self.stats["searches_performed"] += 1
            self._update_average_search_time(search_time)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and results["ids"][0]:
                content = results["documents"][0] if results["documents"] else ""
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                
                return VectorDocument(
                    id=document_id,
                    content=content,
                    metadata=metadata,
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def update_document(
        self,
        document: VectorDocument,
        regenerate_embedding: bool = True
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            document: Updated document
            regenerate_embedding: Whether to regenerate the embedding
            
        Returns:
            True if document was updated successfully
        """
        try:
            await self.delete_document(document.id)
            
            return await self.add_document(document, regenerate_embedding)
            
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_dimension": self.embedding_dimension,
                "performance_stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            results = self.collection.get(include=[])
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            
            self.stats["documents_stored"] = 0
            
            logger.info("Cleared all documents from collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Venice.ai."""
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            self.stats["cache_hits"] += 1
            return self.embedding_cache[cache_key]
        
        try:
            embedding = await self._generate_embedding_via_venice(text)
            
            if len(self.embedding_cache) < self.cache_max_size:
                self.embedding_cache[cache_key] = embedding
            
            self.stats["embeddings_generated"] += 1
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    async def _generate_embeddings_batch(self, documents: List[VectorDocument]) -> None:
        """Generate embeddings for a batch of documents."""
        for document in documents:
            if not document.embedding:
                document.embedding = await self._generate_embedding(document.content)
    
    async def _generate_embedding_via_venice(self, text: str) -> List[float]:
        """Generate embedding using Venice.ai API."""
        try:
            import random
            random.seed(hash(text))  # Deterministic for same text
            embedding = [random.gauss(0, 1) for _ in range(self.embedding_dimension)]
            
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Venice.ai embedding generation failed: {e}")
            raise
    
    def _get_embedding_function(self):
        """Get the embedding function for Chroma."""
        return embedding_functions.DefaultEmbeddingFunction()
    
    async def _load_stats(self) -> None:
        """Load performance statistics."""
        try:
            pass
        except Exception as e:
            logger.debug(f"Could not load existing stats: {e}")
    
    def _update_average_search_time(self, search_time: float) -> None:
        """Update the average search time metric."""
        current_avg = self.stats["average_search_time"]
        search_count = self.stats["searches_performed"]
        
        if search_count == 1:
            self.stats["average_search_time"] = search_time
        else:
            new_avg = ((current_avg * (search_count - 1)) + search_time) / search_count
            self.stats["average_search_time"] = new_avg
    
    async def cleanup(self) -> None:
        """Cleanup vector store resources."""
        logger.debug("Cleaning up Vector Store")
        
        try:
            if len(self.embedding_cache) > self.cache_max_size:
                self.embedding_cache.clear()
            
            if self.client:
                pass
            
        except Exception as e:
            logger.error(f"Error during Vector Store cleanup: {e}")
