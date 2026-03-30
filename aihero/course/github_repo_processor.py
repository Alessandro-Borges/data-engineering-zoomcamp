#!/usr/bin/env python3
"""
GitHub Repository Data Processor and LLM Consultant

# API_KEY
# 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

This script:
1. Fetches markdown documentation from GitHub repositories
2. Processes and chunks the content for efficient search
3. Implements text, vector, and hybrid search capabilities
4. Integrates with OpenAI API for LLM-powered responses

Usage:
    python getfilesrepo.py

Requirements:
    - Set OPENAI_API_KEY environment variable
    - Install dependencies: pip install requests python-frontmatter minsearch sentence-transformers tqdm numpy openai
"""

import os
import io
import zipfile
from typing import List, Dict, Any
import requests
import frontmatter
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.auto import tqdm
import openai


# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        ('requests', 'requests'),
        ('frontmatter', 'frontmatter'),
        ('minsearch', 'minsearch'),
        ('sentence_transformers', 'sentence-transformers'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('openai', 'openai')
    ]

    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with: uv add " + " ".join(missing_packages))
        return False

    print("✅ All dependencies are installed")
    return True


def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print("   PowerShell: $env:OPENAI_API_KEY = 'your-api-key-here'")
        print("   Or permanently in Windows Environment Variables")
        return False

    print("✅ OpenAI API key is configured")
    return True


# =============================================================================
# CONFIGURATION
# =============================================================================

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Model configurations
EMBEDDING_MODEL_NAME = 'multi-qa-distilbert-cos-v1'
LLM_MODEL = 'gpt-4o-mini'

# Search configurations
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 1000
NUM_SEARCH_RESULTS = 5


# =============================================================================
# GITHUB DATA FETCHING
# =============================================================================

def download_github_repo(repo_owner: str, repo_name: str) -> bytes:
    """
    Download a GitHub repository as a ZIP file.

    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name

    Returns:
        ZIP file content as bytes

    Raises:
        Exception: If download fails
    """
    url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to download repository {repo_owner}/{repo_name}: {response.status_code}")

    return response.content


def parse_markdown_files(zip_content: bytes) -> List[Dict[str, Any]]:
    """
    Parse markdown files from a ZIP archive.

    Args:
        zip_content: ZIP file content as bytes

    Returns:
        List of dictionaries containing parsed file data
    """
    repository_data = []

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        for file_info in zf.infolist():
            filename = file_info.filename.lower()

            # Only process markdown files
            if not (filename.endswith('.md') or filename.endswith('.mdx')):
                continue

            try:
                with zf.open(file_info) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    # Parse frontmatter
                    post = frontmatter.loads(content)
                    data = post.to_dict()
                    data['filename'] = file_info.filename
                    repository_data.append(data)
            except Exception as e:
                print(f"Error processing {file_info.filename}: {e}")
                continue

    return repository_data


def read_repo_data(repo_owner: str, repo_name: str) -> List[Dict[str, Any]]:
    """
    Download and parse all markdown files from a GitHub repository.

    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name

    Returns:
        List of dictionaries containing file content and metadata
    """
    zip_content = download_github_repo(repo_owner, repo_name)
    return parse_markdown_files(zip_content)


# =============================================================================
# DATA PROCESSING
# =============================================================================

def sliding_window_chunking(text: str, size: int, step: int) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks using sliding window approach.

    Args:
        text: Input text to chunk
        size: Size of each chunk
        step: Step size for sliding window

    Returns:
        List of chunk dictionaries with start position and content
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    chunks = []
    for i in range(0, len(text), step):
        chunk_text = text[i:i + size]
        if chunk_text:  # Only add non-empty chunks
            chunks.append({
                'start': i,
                'chunk': chunk_text
            })
        if i + size >= len(text):
            break

    return chunks


def process_documents_for_search(documents: List[Dict[str, Any]],
                               chunk_size: int = CHUNK_SIZE,
                               chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Process documents by chunking content and preparing for search indexing.

    Args:
        documents: List of document dictionaries
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents ready for indexing
    """
    processed_chunks = []

    for doc in documents:
        if 'content' not in doc:
            continue

        doc_copy = doc.copy()
        content = doc_copy.pop('content')

        # Create chunks from content
        chunks = sliding_window_chunking(content, chunk_size, chunk_overlap)

        # Add document metadata to each chunk
        for chunk in chunks:
            chunk.update(doc_copy)
            processed_chunks.append(chunk)

    return processed_chunks


# =============================================================================
# SEARCH FUNCTIONALITY
# =============================================================================

def create_text_index(documents: List[Dict[str, Any]], text_fields: List[str]) -> Index:
    """
    Create a text-based search index.

    Args:
        documents: List of document dictionaries
        text_fields: Fields to index for text search

    Returns:
        Configured Index object
    """
    index = Index(text_fields=text_fields, keyword_fields=[])
    index.fit(documents)
    return index


def create_vector_index(documents: List[Dict[str, Any]],
                       embedding_model: SentenceTransformer) -> tuple:
    """
    Create embeddings and vector search index.

    Args:
        documents: List of document dictionaries
        embedding_model: SentenceTransformer model

    Returns:
        Tuple of (embeddings array, VectorSearch index)
    """
    embeddings = []

    for doc in tqdm(documents, desc="Creating embeddings"):
        # Combine relevant text fields for embedding
        text_parts = []
        if 'question' in doc:
            text_parts.append(doc['question'])
        if 'chunk' in doc:
            text_parts.append(doc['chunk'])
        elif 'content' in doc:
            text_parts.append(doc['content'])

        text = ' '.join(text_parts)
        if text.strip():
            embedding = embedding_model.encode(text)
            embeddings.append(embedding)

    embeddings_array = np.array(embeddings)

    vindex = VectorSearch()
    vindex.fit(embeddings_array, documents)

    return embeddings_array, vindex


class DocumentSearch:
    """Combined text and vector search functionality."""

    def __init__(self, text_index: Index, vector_index: VectorSearch,
                 embedding_model: SentenceTransformer):
        self.text_index = text_index
        self.vector_index = vector_index
        self.embedding_model = embedding_model

    def text_search(self, query: str, num_results: int = NUM_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """Perform text-based search."""
        return self.text_index.search(query, num_results=num_results)

    def vector_search(self, query: str, num_results: int = NUM_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """Perform vector-based search."""
        query_embedding = self.embedding_model.encode(query)
        return self.vector_index.search(query_embedding, num_results=num_results)

    def hybrid_search(self, query: str, num_results: int = NUM_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector results."""
        text_results = self.text_search(query, num_results * 2)
        vector_results = self.vector_search(query, num_results * 2)

        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []

        for result in text_results + vector_results:
            doc_id = result.get('filename', str(result))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_results.append(result)
                if len(combined_results) >= num_results:
                    break

        return combined_results


# =============================================================================
# LLM INTEGRATION
# =============================================================================

class LLMConsultant:
    """OpenAI-powered consultant for document-based queries."""

    def __init__(self, api_key: str, model: str = LLM_MODEL):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response using OpenAI API with optional context.

        Args:
            query: User query
            context_documents: Relevant documents for context

        Returns:
            Generated response from LLM
        """
        # Prepare context from documents
        context = ""
        if context_documents:
            context_parts = []
            for i, doc in enumerate(context_documents[:3]):  # Limit to top 3 docs
                title = doc.get('title', doc.get('question', f'Document {i+1}'))
                content = doc.get('chunk', doc.get('content', ''))[:500]  # Limit content length
                context_parts.append(f"Document {i+1} ({title}):\n{content}\n")
            context = "\n".join(context_parts)

        # Create system prompt
        system_prompt = """You are a helpful AI assistant specializing in data engineering and machine learning.
        Answer questions based on the provided context when available. If you don't have relevant information,
        say so clearly."""

        # Create user prompt
        if context:
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_prompt = query

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("🚀 Starting GitHub Repository Data Processor")

    # Check dependencies and configuration
    if not check_dependencies():
        return

    if not check_api_key():
        return

    # Initialize OpenAI client
    llm_consultant = LLMConsultant(OPENAI_API_KEY)

    # Load embedding model
    print("📥 Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Fetch data from repositories
    print("📚 Fetching data from GitHub repositories...")

    # DataTalksClub FAQ
    dtc_faq = read_repo_data('DataTalksClub', 'faq')
    de_dtc_faq = [d for d in dtc_faq if 'data-engineering' in d.get('filename', '')]
    print(f"📄 Loaded {len(de_dtc_faq)} Data Engineering FAQ documents")

    # Evidently documentation
    evidently_docs = read_repo_data('evidentlyai', 'docs')
    evidently_chunks = process_documents_for_search(evidently_docs)
    print(f"📄 Loaded {len(evidently_chunks)} Evidently documentation chunks")

    # Create search indexes
    print("🔍 Creating search indexes...")

    # FAQ text index
    faq_text_index = create_text_index(de_dtc_faq, ["question", "content"])

    # FAQ vector index
    faq_embeddings, faq_vector_index = create_vector_index(de_dtc_faq, embedding_model)

    # Evidently vector index
    evidently_embeddings, evidently_vector_index = create_vector_index(evidently_chunks, embedding_model)

    # Create search objects
    faq_search = DocumentSearch(faq_text_index, faq_vector_index, embedding_model)
    evidently_search = DocumentSearch(None, evidently_vector_index, embedding_model)  # No text index for evidently

    # Example queries
    queries = [
        "What should be in a test dataset for AI evaluation?",
        "Can I join the course now?",
        "How do I monitor model performance?"
    ]

    print("\n" + "="*60)
    print("🤖 LLM CONSULTANT DEMO")
    print("="*60)

    for query in queries:
        print(f"\n❓ Query: {query}")

        # Search for relevant documents
        faq_results = faq_search.hybrid_search(query, num_results=3)
        evidently_results = evidently_search.vector_search(query, num_results=3)

        # Combine results
        all_results = faq_results + evidently_results

        # Generate LLM response
        response = llm_consultant.generate_response(query, all_results)
        print(f"🤖 Response: {response[:200]}..." if len(response) > 200 else f"🤖 Response: {response}")

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()