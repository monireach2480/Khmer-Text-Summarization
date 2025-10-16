# Khmer Text Summarization System

**ITM 454 - Natural Language Processing Final Project**

**Team Members:**
- Len Monireach
- Taing Kimleng
- Seng Sokpanha
- Khvann Munirotha
- Try Chhensorng

---

## 📋 Overview

This project implements a comprehensive extractive text summarization system specifically designed for Khmer language documents. The system uses multiple algorithms and provides comparative analysis of different summarization approaches.

### Key Features
- ✅ **Three Summarization Algorithms**: TextRank, TF-IDF, and Frequency-based
- ✅ **Khmer-Specific Preprocessing**: Custom tokenization and text normalization
- ✅ **ROUGE Evaluation**: Automated quality assessment
- ✅ **Comparative Analysis**: Compare different methods side-by-side
- ✅ **Easy-to-Use API**: Simple interface for quick summarization

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd khmer-text-summarization

# Or simply download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 5: Install KhmerNLTK
```bash
pip install khmernltk
```

---

## 📁 Project Structure

```
khmer-text-summarization/
│
├── khmer_summarization.py      # Main implementation
├── demo_notebook.ipynb          # Jupyter notebook demonstration
├── requirements.txt             # Python dependencies
├── khmer_stopwords.txt          # Khmer stopwords list
├── README.md                    # This file
│
├── data/                        # Dataset directory
│   ├── sample_articles/         # Sample Khmer articles
│   └── reference_summaries/     # Manual summaries for evaluation
│
├── outputs/                     # Generated summaries
│   └── results/                 # Evaluation results
│
└── models/                      # Saved models (if any)
```

---

## 💻 Usage

### Basic Usage

```python
from khmer_summarization import KhmerSummarizationSystem

# Initialize the system
system = KhmerSummarizationSystem()

# Your Khmer text
text = """
ក្នុង​ឱកាស​ទទួល​ ឯកឧត្តម​ផាក​ជុង​វូក (PARK Jung-Wook) 
ឯកអគ្គរដ្ឋទូត​វិសាមញ្ញ​និង​ពេញ​សមត្ថភាព...
"""

# Generate summary
summary = system.summarize(text, method='textrank', num_sentences=3)
print(summary['summary'])
```

### Using Different Methods

```python
# TextRank (graph-based)
textrank_summary = system.summarize(text, method='textrank', num_sentences=3)

# TF-IDF based
tfidf_summary = system.summarize(text, method='tfidf', num_sentences=3)

# Frequency-based (baseline)
freq_summary = system.summarize(text, method='frequency', num_sentences=3)

# Compare all methods
all_summaries = system.summarize_all(text, num_sentences=3)
```

### Using Summary Ratio

```python
# Extract 30% of sentences
summary = system.summarize(text, method='textrank', summary_ratio=0.3)
```

### Loading from File

```python
# Load document
text = system.load_document('article.txt')

# Generate and save summary
summary = system.summarize(text, method='textrank', num_sentences=5)
system.save_summary(summary, 'output_summary.json')
```

### Document Analysis

```python
# Get document statistics
stats = system.analyze_document(text)
print(f"Sentences: {stats['num_sentences']}")
print(f"Words: {stats['num_words']}")
print(f"Unique words: {stats['num_unique_words']}")
```

### Evaluation

```python
# Evaluate summary quality
original_text = "..."
generated_summary = "..."
reference_summary = "..."  # Manual summary (optional)

evaluation = system.evaluate(original_text, generated_summary, reference_summary)
print(f"ROUGE-1: {evaluation['rouge_scores']['rouge-1']['f1']:.3f}")
print(f"ROUGE-2: {evaluation['rouge_scores']['rouge-2']['f1']:.3f}")
print(f"Compression: {evaluation['compression_ratio']:.2%}")
```

---

## 🔬 Algorithms Explained

### 1. TextRank (Primary Method)
- **How it works**: Graph-based algorithm similar to PageRank
- **Process**:
  1. Build sentence similarity matrix
  2. Create graph where sentences are nodes
  3. Apply PageRank to rank sentences
  4. Extract top-ranked sentences
- **Best for**: Most types of documents, balanced performance

### 2. TF-IDF Based
- **How it works**: Ranks sentences by importance of their words
- **Process**:
  1. Calculate TF-IDF scores for all words
  2. Score sentences by sum of word TF-IDF values
  3. Select highest-scoring sentences
- **Best for**: Technical documents, keyword-rich content

### 3. Frequency-Based (Baseline)
- **How it works**: Simple word frequency scoring
- **Process**:
  1. Calculate word frequencies
  2. Score sentences by average word frequency
  3. Select top-scoring sentences
- **Best for**: Quick summaries, baseline comparison

---

## 📊 Evaluation Metrics

### ROUGE Scores
- **ROUGE-1**: Unigram overlap between summary and reference
- **ROUGE-2**: Bigram overlap (captures word order)
- **ROUGE-L**: Longest common subsequence

### Other Metrics
- **Compression Ratio**: Percentage of original sentences retained
- **Coverage**: How much of the original information is preserved

### Expected Performance
Based on Khmer as a low-resource language:
- ROUGE-1 F1: 0.30 - 0.40
- ROUGE-2 F1: 0.12 - 0.20
- ROUGE-L F1: 0.25 - 0.35

---

## 🗂️ Dataset Guidelines

### Data Collection
1. **Sources**:
   - Khmer news websites
   - Official government documents
   - Educational materials
   - Wikipedia articles

2. **Format**: Plain text (.txt) files with UTF-8 encoding

3. **Structure**:
   ```
   data/
   ├── articles/
   │   ├── article_001.txt
   │   ├── article_002.txt
   │   └── ...
   └── references/
       ├── summary_001.txt
       ├── summary_002.txt
       └── ...
   ```

### Data Preprocessing
- Remove headers/footers
- Clean special characters
- Ensure proper UTF-8 encoding
- Verify sentence boundaries

---

## 🧪 Testing

### Run Demo
```bash
python khmer_summarization.py
```

### Run Jupyter Notebook
```bash
jupyter notebook demo_notebook.ipynb
```

### Test with Your Own Data
```python
system = KhmerSummarizationSystem()

# Test with your file
text = system.load_document('your_document.txt')
summary = system.summarize(text, method='textrank', num_sentences=3)
print(summary['summary'])
```

---

## 🐛 Troubleshooting

### Common Issues

**1. KhmerNLTK not found**
```bash
pip install khmer-nltk
```

**2. Unicode encoding errors**
- Ensure files are saved with UTF-8 encoding
- Use `encoding='utf-8'` when reading files

**3. Empty summaries**
- Check if text has proper sentence boundaries (។)
- Verify text is in Khmer script
- Ensure minimum 3-4 sentences in input

**4. Low ROUGE scores**
- Normal for low-resource languages
- Focus on qualitative evaluation
- Check if reference summary is appropriate

---

## 📈 Performance Tips

### For Better Summaries
1. **Input Quality**: Clean, well-formatted text works best
2. **Length**: Works best with documents of 10+ sentences
3. **Method Selection**:
   - Use TextRank for general documents
   - Use TF-IDF for technical content
   - Use Frequency as baseline

### Optimization
- For large documents (>1000 sentences), consider chunking
- Adjust `num_sentences` or `summary_ratio` based on document length
- Experiment with different methods for your use case

---

## 🔮 Future Improvements

Potential enhancements for future versions:
- [ ] Abstractive summarization using transformers
- [ ] Multi-document summarization
- [ ] Query-focused summarization
- [ ] Better sentence reordering
- [ ] Integration with Khmer language models
- [ ] Web interface for easy access
- [ ] Support for other Khmer dialects

---

## 📚 References

1. VietHoang1512, "KhmerNLTK: Natural Language Processing Toolkit for Khmer Language." [Online]. Available: https://github.com/VietHoang1512/khmer-nltk

2. R. Mihalcea and P. Tarau, "TextRank: Bringing order into text," in Proc. EMNLP, 2004.

3. C. D. Manning and H. Schütze, Foundations of Statistical Natural Language Processing. Cambridge, MA: MIT Press, 1999.

4. D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. Pearson, 2019.

5. S. Yath, "awesome-khmer-language: A large collection of Khmer language resources." [Online]. Available: https://github.com/seanghay/awesome-khmer-language

---

## 📝 License

This project is developed for educational purposes as part of ITM 454 - Natural Language Processing course.

---

## 👥 Contact

For questions or issues, please contact:
- **Team Lead**: [Your Email]
- **Course**: ITM 454 - Natural Language Processing
- **Institution**: [Your University]

---

## 🙏 Acknowledgments

- Professor Monyrath Buntoun for guidance and support
- KhmerNLTK developers for the excellent toolkit
- All team members for their contributions
- The Khmer NLP community for resources and inspiration

---

**Last Updated**: September 2025
