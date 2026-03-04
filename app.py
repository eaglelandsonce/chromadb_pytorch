import os
import gradio as gr
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Tuple, Dict
import uuid
import re
from collections import Counter
import math

# Initialize the embedding model (using better sentence-transformers model)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Use a more powerful model for better semantic understanding
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    allow_reset=True
))

# Create or get collection
collection_name = "pdf_documents"
try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "PDF document embeddings"}
    )

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove extra whitespace
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_file) -> List[Tuple[str, int]]:
    """
    Extract text from PDF file and split into semantic chunks.
    Returns list of (text_chunk, page_number) tuples.
    """
    if pdf_file is None:
        return []
    
    try:
        reader = PdfReader(pdf_file.name)
        chunks = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                # Clean the text
                text = clean_text(text)
                
                # Split into larger, more meaningful chunks (1000 chars with 200 char overlap)
                chunk_size = 1000
                overlap = 200
                
                # Try to split on sentence boundaries when possible
                sentences = re.split(r'(?<=[.!?])\s+', text)
                
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append((current_chunk.strip(), page_num))
                        # Start new chunk with overlap
                        words = current_chunk.split()
                        overlap_text = " ".join(words[-20:]) if len(words) > 20 else current_chunk
                        current_chunk = overlap_text + " " + sentence + " "
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), page_num))
        
        return chunks
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

def vectorize_and_store(pdf_file) -> str:
    """
    Process PDF file, create embeddings using PyTorch model, and store in ChromaDB.
    """
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."
    
    try:
        # Extract text chunks from PDF
        chunks = extract_text_from_pdf(pdf_file)
        
        if not chunks:
            return "⚠️ No text could be extracted from the PDF."
        
        # Prepare data for ChromaDB
        texts = [chunk[0] for chunk in chunks]
        page_nums = [chunk[1] for chunk in chunks]
        
        # Generate embeddings using PyTorch model with normalization
        with torch.no_grad():
            embeddings = embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for better cosine similarity
                show_progress_bar=True
            )
        
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Create metadata with source information
        pdf_name = os.path.basename(pdf_file.name)
        metadatas = [
            {
                "source": pdf_name,
                "page": page_num,
                "chunk_id": i
            }
            for i, page_num in enumerate(page_nums)
        ]
        
        # Store in ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Get unique page count
        unique_pages = len(set(page_nums))
        
        return f"✅ Successfully processed and stored {len(chunks)} chunks from '{pdf_name}' ({unique_pages} pages)"
    
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def compute_tfidf_score(query: str, text: str, all_texts: List[str]) -> float:
    """
    Compute TF-IDF score for relevance. Higher score means the terms are more unique/distinctive.
    """
    query_words = set(word.lower() for word in query.split() if len(word) > 2)
    text_words = text.lower().split()
    
    if not query_words:
        return 0.0
    
    score = 0.0
    for word in query_words:
        if word in text_words:
            # TF: frequency in this text
            tf = text_words.count(word) / len(text_words)
            
            # IDF: inverse document frequency
            doc_count = sum(1 for t in all_texts if word in t.lower())
            idf = math.log(len(all_texts) / (doc_count + 1))
            
            score += tf * idf
    
    # Normalize by number of query words
    return min(1.0, score / (len(query_words) + 1))

def get_confidence_level(semantic_score: float, keyword_score: float, tfidf_score: float, margin: float = 0.0) -> Tuple[str, float, str]:
    """
    Calculate confidence level based on semantic similarity, keyword overlap, TF-IDF, and score margin.
    Uses calibrated thresholds for better accuracy.
    Returns: (confidence_level, confidence_score, color_emoji)
    """
    # Combine multiple signals for better calibration
    # 50% semantic, 25% TF-IDF, 25% keyword overlap
    base_confidence = (0.5 * semantic_score + 
                      0.25 * tfidf_score + 
                      0.25 * keyword_score)
    
    # Boost based on margin from next result (0.03+ margin is good)
    if margin > 0.03:
        base_confidence = min(1.0, base_confidence + margin * 0.15)
    
    # Apply confidence calibration - more generous thresholds
    if base_confidence >= 0.70:
        return ("🟢 HIGH", base_confidence, "✅")
    elif base_confidence >= 0.50:
        return ("🟡 MEDIUM", base_confidence, "⚠️")
    elif base_confidence >= 0.30:
        return ("🟠 LOW", base_confidence, "❓")
    else:
        return ("🔴 VERY LOW", base_confidence, "❌")

def keyword_score(query: str, text: str) -> float:
    """Calculate keyword overlap score with importance weighting."""
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'be'}
    
    query_words = set(word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2)
    text_lower = text.lower()
    text_words = set(word for word in text_lower.split() if word not in stop_words and len(word) > 2)
    
    if not query_words:
        return 0.0
    
    # Count exact phrase matches (higher weight)
    phrase_score = 0.0
    for word in query_words:
        if word in text_words:
            # Check if word appears as whole word (not substring)
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                phrase_score += 1
    
    overlap = phrase_score / len(query_words)
    return min(1.0, overlap)

def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search for relevant passages using hybrid semantic + keyword search.
    """
    if not query.strip():
        return "⚠️ Please enter a search query."
    
    try:
        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            return "⚠️ No documents in the database. Please upload a PDF first."
        
        # Generate query embedding using PyTorch model with normalization
        with torch.no_grad():
            query_embedding = embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity
            )
        
        # Search in ChromaDB - get more results for re-ranking
        search_size = min(top_k * 3, count)  # Get 3x results for hybrid re-ranking
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=search_size
        )
        
        # Get all documents for TF-IDF calculation
        all_docs = collection.get(include=['documents'])['documents']
        
        # Re-rank results using multi-signal hybrid scoring
        scored_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            semantic_score = 1 - distance  # Convert distance to similarity
            keyword_score_val = keyword_score(query, doc)
            tfidf_score_val = compute_tfidf_score(query, doc, all_docs)
            
            # Weighted hybrid score: 50% semantic, 25% TF-IDF, 25% keyword
            hybrid_score = (0.5 * semantic_score + 
                          0.25 * tfidf_score_val + 
                          0.25 * keyword_score_val)
            
            scored_results.append((doc, metadata, semantic_score, keyword_score_val, tfidf_score_val, hybrid_score))
        
        # Sort by hybrid score and take top_k
        scored_results.sort(key=lambda x: x[4], reverse=True)
        top_results = scored_results[:top_k]
        
        output = f"🔍 **Top {len(top_results)} results for:** '{query}'\n\n"
        output += "=" * 80 + "\n\n"
        
        for i, (doc, metadata, sem_score, kw_score, tfidf_score, hybrid_score) in enumerate(top_results, start=1):
            # Calculate margin from next result
            margin = 0.0
            if i < len(top_results):
                next_score = top_results[i][5]
                margin = hybrid_score - next_score
            
            # Get confidence level
            confidence_level, confidence_score, emoji = get_confidence_level(
                sem_score, kw_score, tfidf_score, margin
            )
            
            output += f"**Result {i}** {emoji}\n"
            output += f"**Confidence:** {confidence_level} ({confidence_score:.1%})\n"
            output += f"**Relevance Breakdown:**\n"
            output += f"  • Semantic Match: {sem_score:.1%}\n"
            output += f"  • Keyword Match: {kw_score:.1%}\n"
            output += f"  • TF-IDF Score: {tfidf_score:.1%}\n"
            output += f"  • Overall Score: {hybrid_score:.3f}\n"
            output += f"**Source:** {metadata['source']} | **Page:** {metadata['page']}\n"
            output += f"**Text:**\n{doc}\n\n"
            output += "-" * 80 + "\n\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error searching documents: {str(e)}"

def clear_database() -> str:
    """
    Clear all documents from the database.
    """
    try:
        global collection, chroma_client
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "PDF document embeddings"}
        )
        return "✅ Database cleared successfully."
    except Exception as e:
        return f"❌ Error clearing database: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PDF Search with ChromaDB & PyTorch", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📄 PDF Search Application
        ### Powered by PyTorch, ChromaDB, and Gradio
        
        Upload PDF documents and search through them using AI-powered semantic search.
        """
    )
    
    with gr.Tab("📤 Upload PDF"):
        gr.Markdown("### Upload and Process PDF Documents")
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(
                    label="Drag and drop PDF file here",
                    file_types=[".pdf"],
                    type="filepath"
                )
                upload_btn = gr.Button("🚀 Process & Store PDF", variant="primary", size="lg")
            with gr.Column():
                upload_output = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )
        
        upload_btn.click(
            fn=vectorize_and_store,
            inputs=[pdf_input],
            outputs=[upload_output]
        )
    
    with gr.Tab("🔍 Search"):
        gr.Markdown("### Search Uploaded Documents")
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your search query",
                    placeholder="What are you looking for?",
                    lines=2
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of results"
                )
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
            
        search_output = gr.Textbox(
            label="Search Results",
            lines=20,
            interactive=False
        )
        
        search_btn.click(
            fn=search_documents,
            inputs=[query_input, top_k_slider],
            outputs=[search_output]
        )
        
        # Add example queries
        gr.Examples(
            examples=[
                ["What is the main topic discussed?"],
                ["key findings and conclusions"],
                ["methodology and approach"],
            ],
            inputs=query_input
        )
    
    with gr.Tab("🗑️ Database Management"):
        gr.Markdown("### Clear Database")
        gr.Markdown("⚠️ **Warning:** This will delete all uploaded documents from the database.")
        clear_btn = gr.Button("Clear All Documents", variant="stop")
        clear_output = gr.Textbox(label="Status", interactive=False)
        
        clear_btn.click(
            fn=clear_database,
            outputs=[clear_output]
        )
    
    gr.Markdown(
        """
        ---
        **Tech Stack:** PyTorch • ChromaDB • Gradio • Sentence Transformers
        """
    )

if __name__ == "__main__":
    print("Starting PDF Search Application...")
    print(f"Device: {device}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
