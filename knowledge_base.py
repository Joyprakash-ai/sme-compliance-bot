#!/usr/bin/env python3
"""
Knowledge base management for EU regulations.
Handles document ingestion, chunking, and vector store management.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import yaml

class RegulationKnowledgeBase:
    """Manages the vector database of EU regulations."""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Text splitter configuration for legal documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Metadata extraction patterns for EU documents
        self.metadata_patterns = {
            "directive": r"Directive\s+(\d{4}/\d{1,4}/EU)",
            "regulation": r"Regulation\s+\(EU\)\s+No\s+(\d+/\d+)",
            "article": r"Article\s+(\d+[a-z]?)",
            "date": r"(\d{1,2}\s+\w+\s+\d{4})"
        }
    
    def load_regulation_documents(self, docs_dir: str = "./regulations") -> List[Document]:
        """Load and parse regulation documents from markdown files."""
        docs_path = Path(docs_dir)
        documents = []
        
        for md_file in docs_path.glob("*.md"):
            print(f"Loading {md_file.name}...")
            
            # Read markdown file
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert markdown to plain text (preserving structure)
            html = markdown.markdown(content)
            # Simple HTML to text conversion (for demo purposes)
            import re
            text = re.sub(r'<[^>]+>', '', html)
            
            # Extract metadata from filename and content
            metadata = self._extract_metadata(md_file.name, text)
            metadata["source"] = str(md_file)
            metadata["type"] = "regulation"
            
            # Create document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def _extract_metadata(self, filename: str, content: str) -> Dict[str, str]:
        """Extract metadata from filename and content."""
        import re
        
        metadata = {
            "filename": filename,
            "directive": "Unknown",
            "article": "N/A",
            "jurisdiction": "EU",
            "topic": self._infer_topic(filename)
        }
        
        # Try to extract directive number
        directive_match = re.search(r"(\d{4}/\d{1,4}/EU)", content)
        if directive_match:
            metadata["directive"] = directive_match.group(1)
        
        # Try to extract regulation number
        regulation_match = re.search(r"Regulation\s+\(EU\)\s+No\s+(\d+/\d+)", content)
        if regulation_match:
            metadata["directive"] = f"Regulation {regulation_match.group(1)}"
        
        # Extract articles mentioned
        articles = re.findall(r"Article\s+(\d+[a-z]?)", content)
        if articles:
            metadata["articles"] = ", ".join(sorted(set(articles)))
        
        # Extract effective date if present
        date_match = re.search(r"effective\s+from\s+(\d{1,2}\s+\w+\s+\d{4})", content, re.IGNORECASE)
        if date_match:
            metadata["effective_date"] = date_match.group(1)
        
        return metadata
    
    def _infer_topic(self, filename: str) -> str:
        """Infer topic from filename."""
        filename_lower = filename.lower()
        
        if "vat" in filename_lower or "tax" in filename_lower:
            return "Taxation/VAT"
        elif "gdpr" in filename_lower or "privacy" in filename_lower:
            return "Data Privacy"
        elif "employment" in filename_lower or "labor" in filename_lower:
            return "Employment Law"
        elif "consumer" in filename_lower:
            return "Consumer Protection"
        elif "environment" in filename_lower:
            return "Environmental Law"
        elif "digital" in filename_lower:
            return "Digital Services"
        else:
            return "General Regulation"
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding."""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # Create metadata for each chunk
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                # Generate unique ID for the chunk
                chunk_hash = hashlib.md5(
                    f"{doc.metadata['source']}_{i}".encode()
                ).hexdigest()[:10]
                chunk_metadata["hash"] = chunk_hash
                
                chunked_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def create_vector_store(self, documents: List[Document], collection_name: str = "eu_regulations"):
        """Create and persist vector store from documents."""
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=collection_name
        )
        
        # Persist to disk
        vectorstore.persist()
        print(f"Vector store created and persisted to {self.persist_dir}")
        
        # Return stats
        stats = self.get_vector_store_stats(vectorstore)
        return vectorstore, stats
    
    def get_vector_store_stats(self, vectorstore: Chroma) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get collection info
            collection = vectorstore._client.get_collection(
                name=vectorstore._collection.name
            )
            
            stats = {
                "total_documents": collection.count(),
                "embedding_dimension": 1536,  # OpenAI ada-002 dimension
                "persist_directory": str(self.persist_dir),
                "collection_name": vectorstore._collection.name
            }
            
            # Count documents by topic
            topic_counts = {}
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            for metadata in metadatas:
                topic = metadata.get("topic", "Unknown")
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            stats["topics"] = topic_counts
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def query_similarity(self, query: str, k: int = 5) -> List[Dict]:
        """Query the vector store for similar regulations."""
        try:
            # Load existing vector store
            vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            
            # Perform similarity search
            results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content[:300] + "...",
                    "metadata": doc.metadata,
                    "similarity_score": round(score, 3),
                    "directive": doc.metadata.get("directive", "Unknown"),
                    "topic": doc.metadata.get("topic", "Unknown")
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []

def main():
    """Command-line interface for knowledge base management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EU Regulation Knowledge Base Manager")
    parser.add_argument("--init", action="store_true", help="Initialize knowledge base")
    parser.add_argument("--query", "-q", help="Query the knowledge base")
    parser.add_argument("--stats", action="store_true", help="Show knowledge base statistics")
    parser.add_argument("--dir", default="./regulations", help="Regulations directory")
    
    args = parser.parse_args()
    
    kb = RegulationKnowledgeBase()
    
    if args.init:
        print("Initializing knowledge base...")
        documents = kb.load_regulation_documents(args.dir)
        print(f"Loaded {len(documents)} regulation documents")
        
        chunked_docs = kb.chunk_documents(documents)
        print(f"Split into {len(chunked_docs)} chunks")
        
        vectorstore, stats = kb.create_vector_store(chunked_docs)
        print(f"Knowledge base initialized:")
        print(f"  - Total chunks: {stats.get('total_documents', 'N/A')}")
        print(f"  - Topics: {stats.get('topics', {})}")
    
    elif args.query:
        print(f"Querying: {args.query}")
        results = kb.query_similarity(args.query)
        
        print(f"\nFound {len(results)} relevant regulations:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['topic']}] {result['directive']}")
            print(f"   Score: {result['similarity_score']}")
            print(f"   Content: {result['content'][:150]}...")
    
    elif args.stats:
        try:
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=kb.embeddings
            )
            stats = kb.get_vector_store_stats(vectorstore)
            print("Knowledge Base Statistics:")
            for key, value in stats.items():
                if key == "topics":
                    print(f"  {key}:")
                    for topic, count in value.items():
                        print(f"    - {topic}: {count}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    else:
        print("Please specify an action: --init, --query, or --stats")

if __name__ == "__main__":
    main()
