# 📄 PDF Search Application with PyTorch & ChromaDB

A powerful PDF document search application built with PyTorch, ChromaDB, and Gradio. Upload PDF documents, automatically vectorize them using AI embeddings, and perform semantic search to find relevant passages.

## 🌟 Features

- **📤 PDF Upload**: Drag-and-drop interface for easy PDF document upload
- **🧠 AI-Powered Embeddings**: Uses PyTorch and Sentence Transformers for semantic understanding
- **💾 Vector Storage**: ChromaDB for efficient vector storage and retrieval
- **🔍 Semantic Search**: Find relevant passages based on meaning, not just keywords
- **📊 Source Tracking**: Each result includes source document and page number
- **🎨 Modern UI**: Clean Gradio interface with multiple tabs

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will start on `http://localhost:7860`

## 📖 Usage

### 1. Upload PDF Documents
- Go to the "Upload PDF" tab
- Drag and drop your PDF file or click to browse
- Click "Process & Store PDF" to vectorize and store the document

### 2. Search Documents
- Go to the "Search" tab
- Enter your search query in the text box
- Adjust the number of results (1-10)
- Click "Search" to find relevant passages

### 3. Database Management
- Go to the "Database Management" tab
- Clear all documents from the database if needed

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework for embeddings
- **ChromaDB**: Vector database for efficient similarity search
- **Gradio**: Web UI framework
- **Sentence Transformers**: Pre-trained models for text embeddings
- **PyPDF2**: PDF text extraction

## 📝 How It Works

1. **PDF Processing**: Extracts text from uploaded PDFs and splits into manageable chunks (500 characters with 50-character overlap)
2. **Vectorization**: Converts text chunks into high-dimensional vectors using the `all-MiniLM-L6-v2` model
3. **Storage**: Stores vectors in ChromaDB with metadata (source file, page number)
4. **Search**: Converts search queries to vectors and finds the most similar document chunks
5. **Results**: Returns top matching passages with similarity scores and source information

## 🔧 Configuration

The application automatically detects and uses GPU if available, otherwise falls back to CPU.

Key parameters you can modify in `app.py`:
- `chunk_size`: Size of text chunks (default: 500 characters)
- `overlap`: Overlap between chunks (default: 50 characters)
- `embedding_model`: Model used for embeddings (default: 'all-MiniLM-L6-v2')
- `server_port`: Port for the Gradio app (default: 7860)

## 📄 License

MIT License