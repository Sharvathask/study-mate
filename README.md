# 📘 StudyMate - AI-Powered PDF Study Assistant

StudyMate is an intelligent study companion that helps you extract knowledge from PDF documents and generate comprehensive answers to your questions. Built with advanced NLP models and vector search capabilities, it provides context-aware responses tailored to different academic requirements.

## ✨ Features

- **PDF Processing**: Upload and index multiple PDF documents simultaneously
- **Intelligent Q&A**: Ask questions and get detailed answers based on your uploaded content
- **Mark-based Responses**: Get answers tailored to different academic levels (2, 3, 8, or 16 marks)
- **Context Retrieval**: View the exact source content used to generate answers
- **Answer Management**: Save and track your study sessions
- **Quiz Generation**: Auto-generate quiz questions from your documents
- **Progress Tracking**: Monitor your study topics and saved answers

## 🔧 Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)

### Required Dependencies
```bash
pip install transformers sentence-transformers faiss-cpu pymupdf gradio torch requests graphviz
```

Or install all at once:
```bash
pip install -q transformers sentence-transformers faiss-cpu pymupdf gradio torch requests graphviz
```

## 🚀 Quick Start

1. **Clone or download the script**
2. **Install dependencies** using the command above
3. **Run the application**:
    ```bash
    python studymate.py
    ```
4. **Open your browser** and navigate to the provided Gradio URL
5. **Start studying** by uploading PDFs and asking questions!

## 📖 How to Use

### 1. Index PDFs
- Go to the "Index PDFs" tab
- Upload one or multiple PDF files
- Click "Index" to process and store the content
- Wait for the confirmation message

### 2. Ask Questions
- Navigate to the "Q&A" tab
- Type your question in the text box
- Select the appropriate marks level:
    - **2 marks**: Formula + 2 short sentences
    - **3 marks**: Formula + 5-6 key points
    - **8 marks**: Brief structured answer with example
    - **16 marks**: Detailed answer with example and explanation
- Click "Get Answer" to receive your response
- Optionally save the answer using "💾 Save This Answer"

### 3. Generate Quizzes
- Visit the "Quiz" tab
- Click "Conduct Quiz" to generate 5 sample questions
- Questions are automatically created from your indexed content

### 4. Review Saved Content
- Check the "Saved Answers" tab to view all your saved Q&A pairs
- Use the "Tracker" tab to monitor your study progress

## ⚙️ Configuration

### Model Settings
```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Embedding model
FALLBACK_MODEL = "google/flan-t5-small"  # Text generation model
CHUNK_SIZE = 800          # Text chunk size
CHUNK_OVERLAP = 200       # Overlap between chunks
TOP_K = 5                 # Number of relevant chunks to retrieve
```

### Customization Options
- **Change Models**: Modify `EMBEDDING_MODEL_NAME` and `FALLBACK_MODEL` for different AI models
- **Adjust Chunk Size**: Modify `CHUNK_SIZE` and `CHUNK_OVERLAP` for different text processing
- **Search Results**: Change `TOP_K` to retrieve more or fewer relevant sections

## 🏗️ Architecture

### Core Components
1. **PDF Processing**: Uses PyMuPDF to extract text from PDF files
2. **Text Chunking**: Splits documents into overlapping segments for better context
3. **Embedding Generation**: Creates semantic embeddings using SentenceTransformers
4. **Vector Search**: Uses FAISS for efficient similarity search
5. **Answer Generation**: Leverages Hugging Face transformers for text generation
6. **Web Interface**: Built with Gradio for easy interaction

### Data Flow
```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
                                                            ↓
Question Input → Query Embedding → Similarity Search → Context Retrieval
                                                            ↓
Context + Question → Prompt Generation → AI Model → Answer Generation
```

## 🎯 Use Cases

- **Academic Study**: Process textbooks and research papers
- **Exam Preparation**: Generate practice questions and detailed answers
- **Research Assistance**: Quickly find relevant information in large documents
- **Note Taking**: Save important Q&A pairs for future reference
- **Self-Assessment**: Track your learning progress over time

## 🔍 Technical Details

### Supported File Types
- PDF files (.pdf)
- Multiple file upload supported

### AI Models Used
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Text Generation**: `google/flan-t5-small`
- **Vector Database**: FAISS (Facebook AI Similarity Search)

### Performance Considerations
- GPU acceleration available for faster processing
- Chunking strategy optimizes memory usage
- LRU caching for embedding model loading
- Normalized embeddings for better similarity matching

## 🐛 Troubleshooting

### Common Issues

**"❌ No text extracted"**
- Ensure PDFs contain extractable text (not just images)
- Check if PDFs are password-protected

**"⚠ No generator available"**
- Verify all dependencies are installed
- Check internet connection for model downloads

**Slow Performance**
- Consider using GPU acceleration
- Reduce `CHUNK_SIZE` for faster processing
- Use smaller models for lower-end hardware

### Error Handling
The application includes comprehensive error handling for:
- PDF reading failures
- Model loading issues
- Network connectivity problems
- Memory limitations

## 🔄 Updates and Maintenance

### Regular Updates
- Models are cached locally after first download
- No internet required after initial setup
- State persists during session but resets on restart

### Extending Functionality
- Add new question types by modifying the `build_prompt` function
- Integrate different AI models by updating model configurations
- Enhance UI by modifying Gradio components

## 📄 License

This project uses several open-source libraries. Please refer to their respective licenses:
- Transformers (Apache 2.0)
- SentenceTransformers (Apache 2.0)
- FAISS (MIT)
- Gradio (Apache 2.0)
- PyMuPDF (AGPL/Commercial)

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Optimizing performance

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in the console
3. Ensure all dependencies are properly installed
4. Verify PDF files are readable and contain text

---

**Happy Studying with StudyMate! 📚✨**
