import numpy as np
import pandas as pd
import networkx as nx
import re
import json
import os
import glob
from math import sqrt, log
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

# Sklearn for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try importing Khmer NLP with fallback
try:
    from khmernltk import sentence_tokenize, word_tokenize as khmer_word_tokenize
    KHMERNLTK_AVAILABLE = True
    print("✓ khmernltk loaded successfully")
except ImportError:
    KHMERNLTK_AVAILABLE = False
    print("⚠ Warning: khmernltk not installed. Using fallback tokenization.")
    print("  Install with: pip install khmernltk or Install with: pip install khmer-nltk")

import warnings
warnings.filterwarnings('ignore')


class KhmerTextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for Khmer documents
    Uses NLTK utilities and Khmer-specific tokenization
    """
    
    def __init__(self, stopwords_file="khmer_stopwords.txt"):
        """Initialize preprocessor with stopwords"""
        self.stopwords = self.load_stopwords(stopwords_file)
        print(f"Loaded {len(self.stopwords)} Khmer stopwords")
    
    def load_stopwords(self, file_path: str) -> set:
        """Load Khmer stopwords from file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    stopwords = set(line.strip() for line in file if line.strip())
                    return stopwords
            else:
                print(f"Warning: {file_path} not found. Using default stopwords.")
                # Default common Khmer stopwords
                default_stopwords = {
                    "នេះ", "នោះ", "ទាំង", "ដែល", "ជា", "មាន", "និង", "បាន", 
                    "ក្នុង", "ពី", "គឺ", "ដោយ", "ទៅ", "ឬ", "ផង", "ទេ"
                }
                return default_stopwords
        except Exception as e:
            print(f"Error loading stopwords: {e}")
            return set()
    
    def normalize_khmer_text(self, text: str) -> str:
        """Normalize Khmer text - handle Unicode variations and clean text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize Khmer punctuation
        text = re.sub(r'[។]+', '។', text)  # Multiple periods
        text = re.sub(r'[៕]+', '៕', text)  # Multiple section marks
        
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        
        return text.strip()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove Khmer and English punctuation"""
        # Khmer punctuation
        khmer_punct = "។៕៖៙៚៛៝៞៟"
        # English punctuation
        english_punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        
        for punct in khmer_punct + english_punct:
            text = text.replace(punct, " ")
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        if KHMERNLTK_AVAILABLE:
            try:
                sentences = sentence_tokenize(text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                return sentences
            except Exception as e:
                print(f"Sentence tokenization error: {e}. Using fallback.")
        
        # Fallback: split by Khmer period
        sentences = text.split('។')
        sentences = [s.strip() + '។' for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if KHMERNLTK_AVAILABLE:
            try:
                words = khmer_word_tokenize(text)
                words = [w.strip() for w in words if len(w.strip()) > 0]
                return words
            except Exception as e:
                print(f"Word tokenization error: {e}. Using fallback.")
        
        # Fallback: simple space split
        return text.split()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove stopwords from word list"""
        return [word for word in words if word.lower() not in self.stopwords]
    
    def preprocess_text(self, text: str, remove_punct: bool = True, 
                       remove_stops: bool = True, normalize: bool = True) -> List[str]:
        """Complete preprocessing pipeline for text"""
        if normalize:
            text = self.normalize_khmer_text(text)
        
        if remove_punct:
            text = self.remove_punctuation(text)
        
        words = self.tokenize_words(text)
        words = [word.lower() for word in words]
        
        if remove_stops:
            words = self.remove_stopwords(words)
        
        words = [word for word in words if len(word) > 1]
        
        return words
    
    def preprocess_sentences(self, text: str) -> Tuple[List[str], List[List[str]]]:
        """Preprocess document maintaining sentence structure"""
        original_sentences = self.tokenize_sentences(text)
        
        processed_sentences = []
        
        for sentence in original_sentences:
            words = self.preprocess_text(sentence, remove_punct=True, 
                                        remove_stops=True, normalize=True)
            # CRITICAL: Keep ALL sentences to maintain index alignment
            # Use placeholder for empty sentences
            processed_sentences.append(words if words else ['<EMPTY>'])
        
        return original_sentences, processed_sentences
    
    
    def format_summary(self, sentences: List[str]) -> str:
        """
        Standardized summary formatting for Khmer text
        Ensures consistent period handling across all methods
        """
        if not sentences:
            return ""
        
        # Remove trailing periods and spaces from each sentence
        clean_sentences = [s.rstrip('។ ') for s in sentences]
        
        # Join with Khmer period and space
        summary = '។ '.join(clean_sentences)
        
        # Ensure summary ends with period
        if not summary.endswith('។'):
            summary += '។'
        
        # Remove any double/triple periods
        summary = re.sub(r'។+', '។', summary)
        
        return summary


class TextRankSummarizer:
    """TextRank algorithm for extractive summarization"""
    
    def __init__(self, preprocessor: KhmerTextPreprocessor):
        self.preprocessor = preprocessor
    
    def cosine_similarity_vectors(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sqrt(sum(a * a for a in vec1))
        norm2 = sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def sentence_similarity(self, sent1: List[str], sent2: List[str]) -> float:
        """Calculate similarity between two sentences"""
        # Handle empty sentences
        if not sent1 or not sent2 or sent1 == ['<EMPTY>'] or sent2 == ['<EMPTY>']:
            return 0.0
        
        all_words = list(set(sent1 + sent2))
        
        if not all_words:
            return 0.0
        
        vector1 = [sent1.count(word) for word in all_words]
        vector2 = [sent2.count(word) for word in all_words]
        
        return self.cosine_similarity_vectors(vector1, vector2)

    def calculate_tfidf_scores(self, sentences: List[List[str]]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for all words in the document
        
        Args:
            sentences: List of tokenized sentences (list of word lists)
        
        Returns:
            Dictionary mapping word -> TF-IDF score
        """
        # Calculate Document Frequency: how many sentences contain each word
        doc_freq = Counter()
        for sentence in sentences:
            unique_words = set(sentence)
            for word in unique_words:
                doc_freq[word] += 1
        
        total_sentences = len(sentences)
        tfidf_scores = {}
        # Calculate IDF for each word
        # IDF = log(total_sentences / sentences_containing_word)
        for word, freq in doc_freq.items():
            idf = log(total_sentences / freq)
            tfidf_scores[word] = idf
        
        return tfidf_scores

    def sentence_similarity_tfidf(self, sent1: List[str], sent2: List[str], 
                               tfidf_scores: Dict[str, float]) -> float:
        """
        Calculate TF-IDF weighted cosine similarity between two sentences
        
        Args:
            sent1: First sentence (list of words)
            sent2: Second sentence (list of words)
            tfidf_scores: Dictionary of TF-IDF scores for all words
        
        Returns:
            Similarity score between 0 and 1
        """
        if not sent1 or not sent2 or sent1 == ['<EMPTY>'] or sent2 == ['<EMPTY>']:
            return 0.0
        all_words = list(set(sent1 + sent2))
        
        if not all_words:
            return 0.0
        
        # Create TF-IDF weighted vectors
        vector1 = []
        vector2 = []
        
        for word in all_words:
            # Get TF-IDF weight for this word
            tfidf_weight = tfidf_scores.get(word, 0)
            
            # TF (term frequency in sentence) * IDF weight
            tf1 = sent1.count(word)
            tf2 = sent2.count(word)
            
            vector1.append(tf1 * tfidf_weight)
            vector2.append(tf2 * tfidf_weight)
        
        return self.cosine_similarity_vectors(vector1, vector2)
    
    
    def build_similarity_matrix(self, sentences: List[List[str]]) -> np.ndarray:
        """Build sentence similarity matrix for graph construction"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        return similarity_matrix
    
    def build_similarity_matrix_tfidf(self, sentences: List[List[str]]) -> np.ndarray:
        """
        Build sentence similarity matrix using TF-IDF weighted similarity
        
        Args:
            sentences: List of preprocessed sentences
        
        Returns:
            NxN similarity matrix where N is number of sentences
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        # Calculate TF-IDF scores once for the entire document
        tfidf_scores = self.calculate_tfidf_scores(sentences)
        
        # Build similarity matrix
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity_tfidf(
                        sentences[i], sentences[j], tfidf_scores
                    )
        
        return similarity_matrix
    
    
    def summarize(self, text: str, num_sentences: int = 3, 
             summary_ratio: float = None, use_tfidf_weighting: bool = True) -> Dict:
        """
        Generate extractive summary using TextRank
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            summary_ratio: Alternative to num_sentences - fraction of sentences (0-1)
            use_tfidf_weighting: If True, use TF-IDF weighted similarity (default)
                            If False, use original word-count similarity
        
        Returns:
            Dictionary containing summary and metadata
        """
        original_sentences, processed_sentences = self.preprocessor.preprocess_sentences(text)
        
        if len(original_sentences) == 0:
            return {
                'summary': "",
                'method': 'TextRank',
                'weighting': 'tfidf' if use_tfidf_weighting else 'word_count',
                'num_sentences': 0,
                'error': 'No sentences found'
            }
        
        if summary_ratio:
            num_sentences = max(1, int(len(original_sentences) * summary_ratio))
        else:
            num_sentences = min(num_sentences, len(original_sentences))
        
        if len(original_sentences) <= num_sentences:
            summary_text = '។ '.join(original_sentences)
            return {
                'summary': summary_text,
                'method': 'TextRank',
                'weighting': 'tfidf' if use_tfidf_weighting else 'word_count',
                'num_sentences': len(original_sentences),
                'note': 'Document too short, returned original'
            }
        
        # Choose similarity calculation method based on parameter
        if use_tfidf_weighting:
            similarity_matrix = self.build_similarity_matrix_tfidf(processed_sentences)
        else:
            similarity_matrix = self.build_similarity_matrix(processed_sentences)
        
        similarity_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(similarity_graph, max_iter=100)
        
        ranked_sentences = sorted(
            [(scores[i], i, original_sentences[i]) for i in range(len(original_sentences))],
            reverse=True,
            key=lambda x: x[0]
        )
        
        selected_indices = sorted([sent[1] for sent in ranked_sentences[:num_sentences]])
        summary_sentences = [original_sentences[i] for i in selected_indices]
        
        summary_text = self.preprocessor.format_summary(summary_sentences)
        
        return {
            'summary': summary_text,
            'method': 'TextRank',
            'weighting': 'tfidf' if use_tfidf_weighting else 'word_count',
            'num_sentences': num_sentences,
            'total_sentences': len(original_sentences),
            'compression_ratio': num_sentences / len(original_sentences),
            'sentence_scores': {i: scores[i] for i in range(len(original_sentences))}
        }


class TFIDFSummarizer:
    """TF-IDF based extractive summarization"""
    
    def __init__(self, preprocessor: KhmerTextPreprocessor):
        self.preprocessor = preprocessor
        self.vectorizer = None
    
    def summarize(self, text: str, num_sentences: int = 3, 
                 summary_ratio: float = None) -> Dict:
        """Generate summary using TF-IDF sentence scoring"""
        original_sentences, processed_sentences = self.preprocessor.preprocess_sentences(text)
        
        if len(original_sentences) == 0:
            return {
                'summary': "",
                'method': 'TF-IDF',
                'num_sentences': 0,
                'error': 'No sentences found'
            }
        
        if summary_ratio:
            num_sentences = max(1, int(len(original_sentences) * summary_ratio))
        else:
            num_sentences = min(num_sentences, len(original_sentences))
        
        if len(original_sentences) <= num_sentences:
            summary_text = '។ '.join(original_sentences)
            return {
                'summary': summary_text,
                'method': 'TF-IDF',
                'num_sentences': len(original_sentences),
                'note': 'Document too short'
            }
        
        sentence_strings = [' '.join(sent) for sent in processed_sentences]
        
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(sentence_strings)
        
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        ranked_indices = np.argsort(sentence_scores)[::-1]
        
        selected_indices = sorted(ranked_indices[:num_sentences])
        summary_sentences = [original_sentences[i] for i in selected_indices]
        
        # Clean sentences before joining to avoid double periods
        summary_text = self.preprocessor.format_summary(summary_sentences)
        
        return {
            'summary': summary_text,
            'method': 'TF-IDF',
            'num_sentences': num_sentences,
            'total_sentences': len(original_sentences),
            'compression_ratio': num_sentences / len(original_sentences),
            'sentence_scores': {i: float(sentence_scores[i]) for i in range(len(sentence_scores))}
        }


class ClusteringSummarizer:
    """
    Unsupervised clustering-based extractive summarization
    Uses K-means to cluster sentences and selects representative sentences
    """
    
    def __init__(self, preprocessor: KhmerTextPreprocessor):
        self.preprocessor = preprocessor
        self.vectorizer = None
    
    def summarize(self, text: str, num_sentences: int = 3, 
             summary_ratio: float = None) -> Dict:
        """
        Generate summary using K-means clustering
        
        Algorithm:
        1. Convert sentences to TF-IDF vectors (feature extraction)
        2. Cluster sentences using K-means (unsupervised learning)
        3. Select sentence closest to each cluster center
        4. Return selected sentences in original order
        """
        original_sentences, processed_sentences = self.preprocessor.preprocess_sentences(text)
        
        if len(original_sentences) == 0:
            return {
                'summary': "",
                'method': 'Clustering',
                'num_sentences': 0,
                'error': 'No sentences found'
            }
        
        if summary_ratio:
            num_sentences = max(1, int(len(original_sentences) * summary_ratio))
        else:
            num_sentences = min(num_sentences, len(original_sentences))
        
        if len(original_sentences) <= num_sentences:
            clean_sentences = [s.rstrip('។ ') for s in original_sentences]
            summary_text = '។ '.join(clean_sentences)
            if not summary_text.endswith('។'):
                summary_text += '។'
            return {
                'summary': summary_text,
                'method': 'Clustering',
                'num_sentences': len(original_sentences),
                'note': 'Document too short, returned original'
            }
        
        # Convert sentences to strings for vectorization
        sentence_strings = [' '.join(sent) for sent in processed_sentences]
        
        # Step 1: Feature extraction using TF-IDF
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(sentence_strings)
        
        # Step 2: Unsupervised learning - K-means clustering
        # Number of clusters = number of sentences we want
        n_clusters = min(num_sentences, len(original_sentences))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)
        
        # Step 3: Select MOST INFORMATIVE sentence in each cluster (not closest to center)
        selected_indices = []

        for i in range(n_clusters):
            # Find all sentences in this cluster
            cluster_sentences = np.where(kmeans.labels_ == i)[0]
            
            if len(cluster_sentences) == 0:
                continue
            
            # Get vectors for sentences in this cluster
            cluster_vectors = tfidf_matrix[cluster_sentences]
            
            # NEW STRATEGY: Pick sentence with HIGHEST TF-IDF importance
            # (not closest to centroid, which picks "average" boring sentences)
            sentence_importance = np.array(cluster_vectors.sum(axis=1)).flatten()
            
            # Select most important sentence in this cluster
            best_idx_in_cluster = sentence_importance.argmax()
            closest_sentence_idx = cluster_sentences[best_idx_in_cluster]
            
            selected_indices.append(closest_sentence_idx)
        
        # Step 4: Sort by original position and create summary
        selected_indices = sorted(selected_indices)
        summary_sentences = [original_sentences[i] for i in selected_indices]
        
        # Clean sentences to avoid double periods
        summary_text = self.preprocessor.format_summary(summary_sentences)
        
        return {
            'summary': summary_text,
            'method': 'Clustering',
            'algorithm': 'K-means',
            'num_sentences': num_sentences,
            'total_sentences': len(original_sentences),
            'num_clusters': n_clusters,
            'compression_ratio': num_sentences / len(original_sentences),
            'cluster_labels': kmeans.labels_.tolist()
        }


class FrequencySummarizer:
    """Simple frequency-based summarization"""
    
    def __init__(self, preprocessor: KhmerTextPreprocessor):
        self.preprocessor = preprocessor
    
    def calculate_word_frequencies(self, words: List[str]) -> Dict[str, float]:
        """Calculate normalized word frequencies using NLTK"""
        freq_dist = FreqDist(words)
        max_freq = max(freq_dist.values()) if freq_dist else 1
        
        normalized_freq = {word: freq / max_freq for word, freq in freq_dist.items()}
        return normalized_freq
    
    def score_sentence(self, sentence: List[str], word_freq: Dict[str, float]) -> float:
        """Score sentence based on word frequencies"""
        if not sentence:
            return 0.0
        
        score = sum(word_freq.get(word, 0) for word in sentence)
        return score / len(sentence)
    
    def summarize(self, text: str, num_sentences: int = 3, 
                 summary_ratio: float = None) -> Dict:
        """Generate summary using word frequency scoring"""
        original_sentences, processed_sentences = self.preprocessor.preprocess_sentences(text)
        
        if len(original_sentences) == 0:
            return {
                'summary': "",
                'method': 'Frequency',
                'num_sentences': 0,
                'error': 'No sentences found'
            }
        
        if summary_ratio:
            num_sentences = max(1, int(len(original_sentences) * summary_ratio))
        else:
            num_sentences = min(num_sentences, len(original_sentences))
        
        if len(original_sentences) <= num_sentences:
            summary_text = '។ '.join(original_sentences)
            return {
                'summary': summary_text,
                'method': 'Frequency',
                'num_sentences': len(original_sentences),
                'note': 'Document too short'
            }
        
        all_words = [word for sent in processed_sentences for word in sent]
        word_freq = self.calculate_word_frequencies(all_words)
        
        sentence_scores = [
            self.score_sentence(sent, word_freq) 
            for sent in processed_sentences
        ]
        
        ranked_indices = sorted(
            range(len(sentence_scores)), 
            key=lambda i: sentence_scores[i], 
            reverse=True
        )
        
        selected_indices = sorted(ranked_indices[:num_sentences])
        summary_sentences = [original_sentences[i] for i in selected_indices]
        
        summary_text = self.preprocessor.format_summary(summary_sentences)
        
        return {
            'summary': summary_text,
            'method': 'Frequency',
            'num_sentences': num_sentences,
            'total_sentences': len(original_sentences),
            'compression_ratio': num_sentences / len(original_sentences),
            'sentence_scores': {i: sentence_scores[i] for i in range(len(sentence_scores))}
        }


class SummarizationEvaluator:
    """Evaluate summarization quality using ROUGE scores"""
    
    def __init__(self, preprocessor: KhmerTextPreprocessor):
        self.preprocessor = preprocessor
    
    def calculate_rouge_scores(self, reference: str, generated: str) -> Dict:
        """Calculate ROUGE scores"""
        ref_words = self.preprocessor.preprocess_text(reference, remove_stops=False)
        gen_words = self.preprocessor.preprocess_text(generated, remove_stops=False)
        
        if not ref_words or not gen_words:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        
        # ROUGE-1
        ref_unigrams = set(ref_words)
        gen_unigrams = set(gen_words)
        
        overlap_1 = len(ref_unigrams.intersection(gen_unigrams))
        rouge_1_precision = overlap_1 / len(gen_unigrams) if gen_unigrams else 0
        rouge_1_recall = overlap_1 / len(ref_unigrams) if ref_unigrams else 0
        rouge_1_f1 = (2 * rouge_1_precision * rouge_1_recall / 
                     (rouge_1_precision + rouge_1_recall) 
                     if (rouge_1_precision + rouge_1_recall) > 0 else 0)
        
        # ROUGE-2
        ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
        gen_bigrams = set(zip(gen_words[:-1], gen_words[1:]))
        
        overlap_2 = len(ref_bigrams.intersection(gen_bigrams))
        rouge_2_precision = overlap_2 / len(gen_bigrams) if gen_bigrams else 0
        rouge_2_recall = overlap_2 / len(ref_bigrams) if ref_bigrams else 0
        rouge_2_f1 = (2 * rouge_2_precision * rouge_2_recall / 
                     (rouge_2_precision + rouge_2_recall) 
                     if (rouge_2_precision + rouge_2_recall) > 0 else 0)
        
        # ROUGE-L
        def lcs_length(X, Y):
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i-1] == Y[j-1]:
                        L[i][j] = L[i-1][j-1] + 1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            return L[m][n]
        
        lcs_len = lcs_length(ref_words, gen_words)
        rouge_l_precision = lcs_len / len(gen_words) if gen_words else 0
        rouge_l_recall = lcs_len / len(ref_words) if ref_words else 0
        rouge_l_f1 = (2 * rouge_l_precision * rouge_l_recall / 
                     (rouge_l_precision + rouge_l_recall) 
                     if (rouge_l_precision + rouge_l_recall) > 0 else 0)
        
        return {
            'rouge-1': {
                'precision': rouge_1_precision,
                'recall': rouge_1_recall,
                'f1': rouge_1_f1
            },
            'rouge-2': {
                'precision': rouge_2_precision,
                'recall': rouge_2_recall,
                'f1': rouge_2_f1
            },
            'rouge-l': {
                'precision': rouge_l_precision,
                'recall': rouge_l_recall,
                'f1': rouge_l_f1
            }
        }
    
    def compression_ratio(self, original: str, summary: str) -> float:
        """Calculate compression ratio"""
        original_sentences = self.preprocessor.tokenize_sentences(original)
        summary_sentences = self.preprocessor.tokenize_sentences(summary)
        
        if not original_sentences:
            return 0.0
        
        return len(summary_sentences) / len(original_sentences)
    
    def evaluate_summary(self, original: str, summary: str, 
                        reference: str = None) -> Dict:
        """Comprehensive summary evaluation"""
        results = {
            'compression_ratio': self.compression_ratio(original, summary),
            'original_length': len(self.preprocessor.tokenize_sentences(original)),
            'summary_length': len(self.preprocessor.tokenize_sentences(summary))
        }
        
        if reference:
            results['rouge_scores'] = self.calculate_rouge_scores(reference, summary)
        
        return results


class KhmerSummarizationSystem:
    """Main interface for Khmer text summarization"""
    
    def __init__(self, stopwords_file: str = "khmer_stopwords.txt"):
        """Initialize the complete summarization system"""
        print("=" * 70)
        print("Initializing Khmer Summarization System...")
        print("=" * 70)
        
        self.preprocessor = KhmerTextPreprocessor(stopwords_file)
        self.textrank = TextRankSummarizer(self.preprocessor)
        self.tfidf = TFIDFSummarizer(self.preprocessor)
        self.frequency = FrequencySummarizer(self.preprocessor)
        self.clustering = ClusteringSummarizer(self.preprocessor)  # ← NEW!
        self.evaluator = SummarizationEvaluator(self.preprocessor)
        
        print("System initialized successfully!")
        print()

    def summarize(self, text: str, method: str = 'textrank', 
             num_sentences: int = 3, summary_ratio: float = None,
             use_tfidf_weighting: bool = True) -> Dict:
        """Generate summary using specified method"""
        method = method.lower()
        
        if method == 'textrank':
            return self.textrank.summarize(text, num_sentences, summary_ratio, use_tfidf_weighting)
        elif method == 'tfidf':
            return self.tfidf.summarize(text, num_sentences, summary_ratio)
        elif method == 'frequency':
            return self.frequency.summarize(text, num_sentences, summary_ratio)
        elif method == 'clustering':  # ← NEW!
            return self.clustering.summarize(text, num_sentences, summary_ratio)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'textrank', 'tfidf', 'frequency', or 'clustering'")
    
    def summarize_all(self, text: str, num_sentences: int = 3) -> Dict[str, Dict]:
        """Generate summaries using all methods for comparison"""
        return {
            'textrank_tfidf': self.summarize(text, 'textrank', num_sentences, use_tfidf_weighting=True),
            'textrank_basic': self.summarize(text, 'textrank', num_sentences, use_tfidf_weighting=False),
            'clustering': self.summarize(text, 'clustering', num_sentences),  # ← NEW!
            'tfidf': self.summarize(text, 'tfidf', num_sentences),
            'frequency': self.summarize(text, 'frequency', num_sentences)
        }
    
    def evaluate(self, original: str, summary: str, reference: str = None) -> Dict:
        """Evaluate summary quality"""
        return self.evaluator.evaluate_summary(original, summary, reference)
    
    def analyze_document(self, text: str) -> Dict:
        """Analyze document statistics"""
        sentences = self.preprocessor.tokenize_sentences(text)
        words = self.preprocessor.tokenize_words(text)
        processed_words = self.preprocessor.preprocess_text(text)
        
        return {
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_unique_words': len(set(words)),
            'num_processed_words': len(processed_words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def save_summary(self, summary_data: Dict, filename: str):
        """Save summary to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"Summary saved to {filename}")
    
    def load_document(self, filename: str) -> str:
        """Load document from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()


def demo_system():
    """Demonstration of the Khmer summarization system (writes output to file)"""
    output_lines = []
    
    output_lines.append("=" * 70)
    output_lines.append("KHMER TEXT SUMMARIZATION SYSTEM - DEMO")
    output_lines.append("=" * 70)
    output_lines.append("")
    
    # Initialize system
    system = KhmerSummarizationSystem()
    
    # Check for articles directory
    articles_dir = os.path.join("data", "sample_articles")
    results_dir = os.path.join("outputs", "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(articles_dir):
        output_lines.append(f"X Directory '{articles_dir}' not found.")
        output_lines.append("Please create the directory and add sample articles.")
        write_results(results_dir, "error_results.txt", output_lines)
        return
    
    article_files = glob.glob(os.path.join(articles_dir, "*.txt"))
    if not article_files:
        output_lines.append(f"X No .txt files found in '{articles_dir}'")
        output_lines.append("Please add at least one Khmer text file.")
        write_results(results_dir, "error_results.txt", output_lines)
        return
    
    # Load the first article
    article_file = article_files[0]
    article_name = os.path.basename(article_file)
    
    output_lines.append(f"Loading article: {article_name}")
    output_lines.append("-" * 70)
    
    try:
        sample_text = system.load_document(article_file)
        output_lines.append(f"Successfully loaded: {article_name}")
        output_lines.append("")
    except Exception as e:
        output_lines.append(f"Error loading article: {e}")
        write_results(results_dir, f"{article_name}_results.txt", output_lines)
        return
    
    # Document analysis
    output_lines.append("1. DOCUMENT ANALYSIS")
    output_lines.append("-" * 70)
    doc_stats = system.analyze_document(sample_text)
    output_lines.append(f"Sentences: {doc_stats['num_sentences']}")
    output_lines.append(f"Words: {doc_stats['num_words']}")
    output_lines.append(f"Unique words: {doc_stats['num_unique_words']}")
    output_lines.append(f"Average sentence length: {doc_stats['avg_sentence_length']:.1f} words")
    output_lines.append("")
    
    # Summarization
    output_lines.append("2. SUMMARIZATION RESULTS")
    output_lines.append("-" * 70)
    summaries = system.summarize_all(sample_text, num_sentences=2)
    
    for method, result in summaries.items():
        output_lines.append(f"\n{method.upper()} Method:")
        summary_preview = result['summary']
        if len(summary_preview) > 200:
            summary_preview = summary_preview[:200] + "..."
        output_lines.append(f"Summary: {summary_preview}")
        output_lines.append(f"Compression: {result.get('compression_ratio', 0):.2%}")
    
    output_lines.append("")
    
    # COMPARATIVE EVALUATION - All methods
    output_lines.append("3. COMPARATIVE EVALUATION")
    output_lines.append("-" * 70)
    
    # Check if reference exists
    references_dir = os.path.join("data", "reference_summaries")
    base_name = os.path.splitext(article_name)[0]
    reference_file = os.path.join(references_dir, f"{base_name}.txt")
    
    if os.path.exists(reference_file):
        try:
            reference_summary = system.load_document(reference_file)
            
            output_lines.append("\nROUGE Scores Comparison:")
            output_lines.append("-" * 70)
            output_lines.append(f"{'Method':<20} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
            output_lines.append("-" * 70)
            
            # Store results for finding best method
            results_comparison = {}
            
            # Evaluate each method
            for method_name, summary_result in summaries.items():
                evaluation = system.evaluate(
                    sample_text, 
                    summary_result['summary'], 
                    reference_summary
                )
                
                if 'rouge_scores' in evaluation:
                    rouge_1 = evaluation['rouge_scores']['rouge-1']['f1']
                    rouge_2 = evaluation['rouge_scores']['rouge-2']['f1']
                    rouge_l = evaluation['rouge_scores']['rouge-l']['f1']
                    
                    results_comparison[method_name] = {
                        'rouge-1': rouge_1,
                        'rouge-2': rouge_2,
                        'rouge-l': rouge_l
                    }
                    
                    output_lines.append(
                        f"{method_name:<20} {rouge_1:<10.3f} {rouge_2:<10.3f} {rouge_l:<10.3f}"
                    )
            
            output_lines.append("-" * 70)
            
            # Find and display best method for each metric
            if results_comparison:
                output_lines.append("")
                output_lines.append("Best Performing Method per Metric:")
                output_lines.append("-" * 70)
                
                # Find best for ROUGE-1
                best_r1 = max(results_comparison.items(), key=lambda x: x[1]['rouge-1'])
                output_lines.append(f"ROUGE-1: {best_r1[0]} (F1={best_r1[1]['rouge-1']:.3f})")
                
                # Find best for ROUGE-2
                best_r2 = max(results_comparison.items(), key=lambda x: x[1]['rouge-2'])
                output_lines.append(f"ROUGE-2: {best_r2[0]} (F1={best_r2[1]['rouge-2']:.3f})")
                
                # Find best for ROUGE-L
                best_rl = max(results_comparison.items(), key=lambda x: x[1]['rouge-l'])
                output_lines.append(f"ROUGE-L: {best_rl[0]} (F1={best_rl[1]['rouge-l']:.3f})")
                
                # Calculate average and find overall best
                output_lines.append("")
                output_lines.append("Average ROUGE Scores (across all metrics):")
                output_lines.append("-" * 70)
                
                avg_scores = {}
                for method_name, scores in results_comparison.items():
                    avg = (scores['rouge-1'] + scores['rouge-2'] + scores['rouge-l']) / 3
                    avg_scores[method_name] = avg
                    output_lines.append(f"{method_name:<20} {avg:.3f}")
                
                best_overall = max(avg_scores.items(), key=lambda x: x[1])
                output_lines.append("-" * 70)
                output_lines.append(f"** BEST OVERALL: {best_overall[0]} (Avg={best_overall[1]:.3f}) **")
            
            output_lines.append("")
            
        except Exception as e:
            output_lines.append(f"Error in comparative evaluation: {e}")
    else:
        output_lines.append("\nNo reference summary found for ROUGE evaluation")
        output_lines.append(f"Expected at: {reference_file}")
        output_lines.append("Create a reference summary to enable method comparison.")
        
        # Still show basic evaluation without ROUGE
        output_lines.append("")
        output_lines.append("Basic Evaluation (without reference):")
        output_lines.append("-" * 70)
        for method_name, summary_result in summaries.items():
            evaluation = system.evaluate(sample_text, summary_result['summary'])
            output_lines.append(f"{method_name:<20} Compression: {evaluation['compression_ratio']:.2%}")
    
    output_lines.append("")
    output_lines.append("=" * 70)
    
    # Write all results to file
    result_filename = f"{base_name}_results.txt"
    write_results(results_dir, result_filename, output_lines)
    print(f"Results saved to: {os.path.join(results_dir, result_filename)}")


def write_results(directory, filename, lines):
    """Helper to write output lines to a file."""
    filepath = os.path.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    demo_system()