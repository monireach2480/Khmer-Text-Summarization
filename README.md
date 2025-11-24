# Khmer Text Summarization System

**ITM 454 - Natural Language Processing Final Project**

## Team Members

- Len Monireach
- Taing Kimleng
- Seng Sokpanha
- Khvann Munirotha
- Try Chhensorng

---

![Khmer Text Summarization Demo 1](./images/imagee.png)
![Khmer Text Summarization Demo 2](./images/imageee.png)
![Khmer Text Summarization Demo 3](./images/imageeee.png)


## 📋 Overview

This project implements a comprehensive extractive text summarization system specifically designed for Khmer language documents. The system uses multiple algorithms including advanced TextRank with TF-IDF weighting and clustering-based approaches, providing comparative analysis of different summarization methods.

### Key Features

✅ **Four Summarization Algorithms**: TextRank (with TF-IDF weighting), TF-IDF, Frequency-based, and Clustering

✅ **Advanced TextRank**: TF-IDF weighted similarity for improved performance

✅ **Unsupervised Clustering**: K-means based sentence selection

✅ **Khmer-Specific Preprocessing**: Custom tokenization and text normalization

✅ **ROUGE Evaluation**: Automated quality assessment with comparative analysis

✅ **Web Interface**: Streamlit-based user-friendly application

✅ **Comparative Analysis**: Compare different methods side-by-side with performance metrics

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
├── app.py                      # Streamlit web application
├── demo_notebook.ipynb         # Jupyter notebook demonstration
├── requirements.txt            # Python dependencies
├── khmer_stopwords.txt         # Khmer stopwords list
├── README.md                   # This file
│
├── data/                       # Dataset directory
│   ├── sample_articles/        # Sample Khmer articles
│   └── reference_summaries/    # Manual summaries for evaluation
│
├── outputs/                    # Generated summaries
│   └── results/                # Evaluation results
│
└── models/                     # Saved models (if any)
```

---

## 💻 Usage

### Web Application (Recommended)

```bash
streamlit run app.py
```

The web interface provides:
- Real-time summarization
- Method comparison
- Visual analytics
- ROUGE score evaluation
- Downloadable results

### Basic Python Usage

```python
from khmer_summarization import KhmerSummarizationSystem

# Initialize the system
system = KhmerSummarizationSystem()

# Your Khmer text
text = """
ក្នុង​ឱកាស​ទទួល​ ឯកឧត្តម​ផាក​ជុង​វូក (PARK Jung-Wook) 
ឯកអគ្គរដ្ឋទូត​វិសាមញ្ញ​និង​ពេញ​សមត្ថភាព...
"""

# Generate summary with advanced TextRank
summary = system.summarize(text, method='textrank', num_sentences=3, use_tfidf_weighting=True)
print(summary['summary'])
```

### Using Different Methods

```python
# TextRank with TF-IDF weighting (recommended)
textrank_tfidf = system.summarize(text, method='textrank', num_sentences=3, use_tfidf_weighting=True)

# TextRank with basic word count
textrank_basic = system.summarize(text, method='textrank', num_sentences=3, use_tfidf_weighting=False)

# Clustering-based (new!)
clustering_summary = system.summarize(text, method='clustering', num_sentences=3)

# TF-IDF based
tfidf_summary = system.summarize(text, method='tfidf', num_sentences=3)

# Frequency-based (baseline)
freq_summary = system.summarize(text, method='frequency', num_sentences=3)

# Compare all methods with performance metrics
all_summaries = system.summarize_all(text, num_sentences=3)
```

### Using Summary Ratio

```python
# Extract 30% of sentences
summary = system.summarize(text, method='textrank', summary_ratio=0.3, use_tfidf_weighting=True)
```

### Loading from File

```python
# Load document
text = system.load_document('article.txt')

# Generate and save summary
summary = system.summarize(text, method='textrank', num_sentences=5, use_tfidf_weighting=True)
system.save_summary(summary, 'output_summary.json')
```

### Document Analysis

```python
# Get document statistics
stats = system.analyze_document(text)
print(f"Sentences: {stats['num_sentences']}")
print(f"Words: {stats['num_words']}")
print(f"Unique words: {stats['num_unique_words']}")
print(f"Average sentence length: {stats['avg_sentence_length']:.1f}")
```

### Evaluation

```python
# Evaluate summary quality
original_text = "..."
generated_summary = "..."
reference_summary = "..."  # Manual summary (optional)

evaluation = system.evaluate(original_text, generated_summary, reference_summary)
print(f"ROUGE-1 F1: {evaluation['rouge_scores']['rouge-1']['f1']:.3f}")
print(f"ROUGE-2 F1: {evaluation['rouge_scores']['rouge-2']['f1']:.3f}")
print(f"ROUGE-L F1: {evaluation['rouge_scores']['rouge-l']['f1']:.3f}")
print(f"Compression: {evaluation['compression_ratio']:.2%}")
```

---

## 🔬 Algorithms Explained

### 1. TextRank with TF-IDF Weighting (Primary Method)

**How it works**: Enhanced graph-based algorithm with TF-IDF weighted similarity

**Process**:
1. Calculate TF-IDF scores for all words in document
2. Build sentence similarity matrix using TF-IDF weighted cosine similarity
3. Create graph where sentences are nodes and similarities are edges
4. Apply PageRank to rank sentences by importance
5. Extract top-ranked sentences in original order

**Best for**: All document types, provides most balanced and accurate results

### 2. Clustering-Based (New!)

**How it works**: Unsupervised learning approach using K-means clustering

**Process**:
1. Convert sentences to TF-IDF vectors
2. Cluster sentences using K-means (number of clusters = desired summary length)
3. Select most informative sentence from each cluster
4. Return selected sentences in original order

**Best for**: Diverse documents, ensures coverage of different topics

### 3. TF-IDF Based

**How it works**: Ranks sentences by importance of their words using TF-IDF

**Process**:
1. Calculate TF-IDF scores for all words
2. Score sentences by sum of word TF-IDF values
3. Select highest-scoring sentences

**Best for**: Technical documents, keyword-rich content

### 4. Frequency-Based (Baseline)

**How it works**: Simple word frequency scoring using NLTK

**Process**:
1. Calculate word frequencies using NLTK FreqDist
2. Score sentences by average word frequency
3. Select top-scoring sentences

**Best for**: Quick summaries, baseline comparison

---

## 📊 Evaluation Metrics

### ROUGE Scores

- **ROUGE-1**: Unigram overlap between summary and reference
- **ROUGE-2**: Bigram overlap (captures word order and phrases)
- **ROUGE-L**: Longest common subsequence (captures sentence structure)

### Other Metrics

- **Compression Ratio**: Percentage of original sentences retained
- **Coverage**: How much of the original information is preserved
- **Vocabulary Diversity**: Ratio of unique words to total words
- **Method Performance**: Comparative analysis across all algorithms

### Expected Performance

Based on enhanced algorithms for Khmer language:
- **ROUGE-1 F1**: 0.35 - 0.50 (improved with TF-IDF weighting)
- **ROUGE-2 F1**: 0.18 - 0.30 (improved with better similarity measures)
- **ROUGE-L F1**: 0.30 - 0.45 (better sentence structure preservation)

---

## 🗂️ Dataset Guidelines

### Data Collection

**Sources**:
- Khmer news websites
- Official government documents
- Educational materials
- Wikipedia articles
- Academic papers in Khmer

**Format**: Plain text (.txt) files with UTF-8 encoding

**Structure**:
```
data/
├── sample_articles/
│   ├── article_001.txt
│   ├── article_002.txt
│   └── ...
└── reference_summaries/
    ├── article_001.txt
    ├── article_002.txt
    └── ...
```

### Data Preprocessing

- Remove headers/footers
- Clean special characters while preserving Khmer punctuation
- Ensure proper UTF-8 encoding
- Verify sentence boundaries using Khmer period (។)
- Normalize Unicode variations

---

## 🧪 Testing

### Run Web Application

```bash
streamlit run app.py
```

### Run Demo Script

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

# Try different methods
summary_tfidf = system.summarize(text, method='textrank', num_sentences=3, use_tfidf_weighting=True)
summary_cluster = system.summarize(text, method='clustering', num_sentences=3)

print("TF-IDF Weighted TextRank:", summary_tfidf['summary'])
print("Clustering Method:", summary_cluster['summary'])
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. KhmerNLTK not found

```bash
pip install khmernltk
# or alternative
pip install khmer-nltk
```

#### 2. Unicode encoding errors

- Ensure files are saved with UTF-8 encoding
- Use `encoding='utf-8'` when reading files
- Check for zero-width characters and normalize text

#### 3. Empty summaries

- Check if text has proper sentence boundaries (។)
- Verify text is in Khmer script
- Ensure minimum 3-4 sentences in input
- Check stopwords file exists

#### 4. Low ROUGE scores

- Normal for low-resource languages but improved with new algorithms
- Focus on qualitative evaluation
- Check if reference summary is appropriate
- Try TF-IDF weighted TextRank for better performance

#### 5. Streamlit app not starting

```bash
pip install streamlit
streamlit run app.py
```

---

## 📈 Performance Tips

### For Better Summaries

- **Input Quality**: Clean, well-formatted text with proper Khmer punctuation
- **Document Length**: Works best with documents of 10-50 sentences
- **Method Selection**:
  - Use TextRank with TF-IDF weighting for general documents (recommended)
  - Use Clustering for documents with multiple topics
  - Use TF-IDF for technical or keyword-rich content
  - Use Frequency as baseline reference

### Optimization

- For large documents (>100 sentences), consider chunking
- Adjust `num_sentences` or `summary_ratio` based on document complexity
- Use the web app for visual comparison of different methods
- Enable TF-IDF weighting in TextRank for improved accuracy

### Web App Features

- Real-time method comparison
- Visual analytics and score charts
- Download results in JSON format
- History tracking of previous summaries
- Interactive ROUGE evaluation

---

## 🔮 Future Improvements

Potential enhancements for future versions:
- Abstractive summarization using transformers
- Khmer BERT integration for better embeddings
- Multi-document summarization
- Query-focused summarization
- Better sentence reordering and coherence
- Integration with larger Khmer language models
- Mobile application interface
- Support for other Khmer dialects and historical texts
- Real-time summarization API

---

## 📚 References

1. VietHoang1512, "KhmerNLTK: Natural Language Processing Toolkit for Khmer Language." [Online]. Available: https://github.com/VietHoang1512/khmer-nltk
2. R. Mihalcea and P. Tarau, "TextRank: Bringing order into text," in Proc. EMNLP, 2004.
3. C. D. Manning and H. Schütze, Foundations of Statistical Natural Language Processing. Cambridge, MA: MIT Press, 1999.
4. D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. Pearson, 2019.
5. S. Yath, "awesome-khmer-language: A large collection of Khmer language resources." [Online]. Available: https://github.com/seanghay/awesome-khmer-language
6. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.



## 🙏 Acknowledgments

- Professor Monyrath BUNTOUN for guidance and support
- KhmerNLTK developers for the excellent toolkit
- Scikit-learn team for machine learning utilities
- Streamlit team for the web application framework
- All team members for their contributions and enhancements
- The Khmer NLP community for resources and inspiration