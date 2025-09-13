# Khmer News Summarizer - ITM 454 Final Project

This project is a functional prototype of an extractive text summarizer for news articles written in the Khmer language. It addresses the challenge of information overload by providing concise summaries of long articles, taking into account the unique linguistic features of Khmer, such as word segmentation.

## Features

-   **Web Scraping**: Fetches live news articles directly from a URL (specifically tailored for `vodkhmer.news`).
-   **Khmer-Specific Preprocessing**: A robust pipeline that handles sentence segmentation, word tokenization, and stopword removal for Khmer text.
-   **Extractive Summarization**: Implements the TextRank algorithm using TF-IDF vectors to identify and extract the most important sentences from an article.
-   **Evaluation**: Calculates ROUGE scores to measure the quality of the summary against a baseline.

## Technical Stack

-   **Language**: Python 3.8+
-   **Core Libraries**:
    -   `Jupyter Notebook`: For interactive development and demonstration.
    -   `requests` & `BeautifulSoup4`: For web scraping.
    -   `khmernltk`: For accurate Khmer word tokenization.
    -   `scikit-learn`: For TF-IDF vectorization.
    -   `networkx`: For building the graph for the TextRank algorithm.
    -   `rouge-score`: For summary evaluation.
    -   `nltk`: For sentence tokenization.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open the Notebook:**
    In the Jupyter interface, open the `Khmer_News_Summarizer.ipynb` file.

3.  **Run the Cells:**
    You can run all the cells sequentially to see the entire process, from scraping a live article to generating and evaluating its summary. The final section provides a clear demonstration.

---