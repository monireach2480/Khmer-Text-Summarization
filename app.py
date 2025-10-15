import streamlit as st
import sys
import os
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import your summarization system
from khmer_summarization import (
    KhmerTextPreprocessor,
    TextRankSummarizer,
    TFIDFSummarizer,
    FrequencySummarizer,
    SummarizationEvaluator,
    KhmerSummarizationSystem
)

# Page configuration
st.set_page_config(
    page_title="Khmer Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --background-color: #F7F9FC;
        --text-color: #2C3E50;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    /* Card styling */
    .summary-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: #2C3E50;  /* Add this line */
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 5px;
        border: 2px solid #e0e0e0;
        font-family: 'Khmer OS Battambang', Arial, sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    with st.spinner('Loading Khmer Summarization System...'):
        st.session_state.system = KhmerSummarizationSystem()
        st.session_state.history = []

def create_metric_card(value, label, color_start, color_end):
    """Create a metric card with gradient background"""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
            {value}
        </div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            {label}
        </div>
    </div>
    """

def plot_rouge_scores(rouge_scores):
    """Create a beautiful bar chart for ROUGE scores"""
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    f1_scores = [
        rouge_scores['rouge-1']['f1'],
        rouge_scores['rouge-2']['f1'],
        rouge_scores['rouge-l']['f1']
    ]
    precision_scores = [
        rouge_scores['rouge-1']['precision'],
        rouge_scores['rouge-2']['precision'],
        rouge_scores['rouge-l']['precision']
    ]
    recall_scores = [
        rouge_scores['rouge-1']['recall'],
        rouge_scores['rouge-2']['recall'],
        rouge_scores['rouge-l']['recall']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='F1', x=metrics, y=f1_scores, marker_color='#667eea'))
    fig.add_trace(go.Bar(name='Precision', x=metrics, y=precision_scores, marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(name='Recall', x=metrics, y=recall_scores, marker_color='#FF6B6B'))
    
    fig.update_layout(
        title='ROUGE Evaluation Scores',
        barmode='group',
        template='plotly_white',
        height=400,
        font=dict(size=12)
    )
    
    return fig

def plot_sentence_scores(scores_dict, method_name):
    """Create a line chart for sentence scores"""
    indices = list(scores_dict.keys())
    scores = list(scores_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices,
        y=scores,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    ))
    
    fig.update_layout(
        title=f'Sentence Importance Scores ({method_name})',
        xaxis_title='Sentence Index',
        yaxis_title='Score',
        template='plotly_white',
        height=350
    )
    
    return fig

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📝 Khmer Text Summarizer</h1>
            <p>Advanced Extractive Summarization for Khmer Documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_Cambodia.svg", width=100)
        st.title("⚙️ Configuration")
        
        # Method selection
        method = st.selectbox(
            "Summarization Method",
            ["TextRank", "TF-IDF", "Frequency"],
            help="Choose the algorithm for summarization"
        )
        
        # Summary length
        summary_type = st.radio(
            "Summary Length",
            ["Number of Sentences", "Ratio"],
            help="Specify summary length by sentences or percentage"
        )
        
        if summary_type == "Number of Sentences":
            num_sentences = st.slider("Number of Sentences", 1, 10, 3)
            summary_ratio = None
        else:
            summary_ratio = st.slider("Summary Ratio", 0.1, 0.5, 0.3, 0.05)
            num_sentences = None
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_scores = st.checkbox("Show Sentence Scores", value=True)
            show_stats = st.checkbox("Show Document Statistics", value=True)
            compare_methods = st.checkbox("Compare All Methods", value=False)
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Khmer Text Summarizer**
            
            This tool uses advanced NLP techniques to automatically 
            generate summaries of Khmer text documents.
            
            **Methods:**
            - **TextRank**: Graph-based ranking
            - **TF-IDF**: Term frequency analysis
            - **Frequency**: Word frequency scoring
            
            **Built with:** Python, NLTK, NetworkX, Streamlit
            """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Summarize", "📊 Analysis", "📚 History"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Input Text")
            
            # Input options
            input_method = st.radio(
                "Input Method",
                ["Text Input", "Upload File"],
                horizontal=True
            )
            
            if input_method == "Text Input":
                input_text = st.text_area(
                    "Enter Khmer text to summarize",
                    height=400,
                    placeholder='អត្ថបទខ្មែរឧទាហរណា៖ សូមវាយបញ្ចូលអត្ថបទនៅទីនេះ...',
                    help="Paste or type your Khmer text here"
                )

            else:
                uploaded_file = st.file_uploader(
                    "Upload a text file",
                    type=['txt'],
                    help="Upload a .txt file containing Khmer text"
                )
                
                if uploaded_file:
                    input_text = uploaded_file.read().decode('utf-8')
                    st.text_area("File Content", input_text, height=400, disabled=True)
                else:
                    input_text = ""
            
            # Summarize button
            if st.button("🚀 Generate Summary", use_container_width=True):
                if input_text.strip():
                    with st.spinner('Generating summary...'):
                        try:
                            # Generate summary
                            if compare_methods:
                                summaries = st.session_state.system.summarize_all(
                                    input_text, 
                                    num_sentences=num_sentences or 3
                                )
                                st.session_state.current_summaries = summaries
                                st.session_state.current_text = input_text
                            else:
                                summary = st.session_state.system.summarize(
                                    input_text,
                                    method=method.lower(),
                                    num_sentences=num_sentences,
                                    summary_ratio=summary_ratio
                                )
                                st.session_state.current_summary = summary
                                st.session_state.current_text = input_text
                                st.session_state.current_method = method
                            
                            st.success("✅ Summary generated successfully!")
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'method': method if not compare_methods else "All Methods",
                                'text_length': len(input_text),
                                'input_text': input_text[:200] + "..."
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter some text to summarize")
        
        with col2:
            st.markdown("### Summary Results")
            
            if compare_methods and 'current_summaries' in st.session_state:
                summaries = st.session_state.current_summaries
                
                for method_name, result in summaries.items():
                    with st.expander(f"📌 {method_name.upper()} Summary", expanded=True):
                        st.markdown(f"""
                        <div class="summary-card">
                            <p style="line-height: 1.8; font-size: 1.05rem;">
                                {result['summary']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Sentences", result['num_sentences'])
                        with metric_col2:
                            compression = result.get('compression_ratio', 0)
                            st.metric("Compression", f"{compression:.1%}")
                        
                        if show_scores and 'sentence_scores' in result:
                            st.plotly_chart(
                                plot_sentence_scores(result['sentence_scores'], method_name),
                                use_container_width=True
                            )
            
            elif 'current_summary' in st.session_state:
                summary = st.session_state.current_summary
                
                st.markdown(f"""
                <div class="summary-card">
                    <h4 style="color: #667eea; margin-bottom: 1rem;">
                        {st.session_state.current_method} Summary
                    </h4>
                    <p style="line-height: 1.8; font-size: 1.05rem;">
                        {summary['summary']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics in cards
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown(
                        create_metric_card(
                            summary['num_sentences'],
                            "Sentences",
                            "#f093fb",
                            "#f5576c"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col_m2:
                    st.markdown(
                        create_metric_card(
                            summary['total_sentences'],
                            "Total Sentences",
                            "#4facfe",
                            "#00f2fe"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col_m3:
                    compression = summary.get('compression_ratio', 0)
                    st.markdown(
                        create_metric_card(
                            f"{compression:.1%}",
                            "Compression",
                            "#43e97b",
                            "#38f9d7"
                        ),
                        unsafe_allow_html=True
                    )
                
                # Sentence scores visualization
                if show_scores and 'sentence_scores' in summary:
                    st.markdown("---")
                    st.plotly_chart(
                        plot_sentence_scores(
                            summary['sentence_scores'],
                            st.session_state.current_method
                        ),
                        use_container_width=True
                    )
                
                # Download button
                st.markdown("---")
                summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Download Summary (JSON)",
                    data=summary_json,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            else:
                st.info("👈 Enter text and click 'Generate Summary' to see results")
    
    with tab2:
        st.markdown("### 📊 Document Analysis")
        
        if 'current_text' in st.session_state:
            text = st.session_state.current_text
            
            # Analyze document
            stats = st.session_state.system.analyze_document(text)
            
            # Display statistics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.markdown(
                    create_metric_card(
                        stats['num_sentences'],
                        "Sentences",
                        "#667eea",
                        "#764ba2"
                    ),
                    unsafe_allow_html=True
                )
            
            with col_s2:
                st.markdown(
                    create_metric_card(
                        stats['num_words'],
                        "Words",
                        "#f093fb",
                        "#f5576c"
                    ),
                    unsafe_allow_html=True
                )
            
            with col_s3:
                st.markdown(
                    create_metric_card(
                        stats['num_unique_words'],
                        "Unique Words",
                        "#4facfe",
                        "#00f2fe"
                    ),
                    unsafe_allow_html=True
                )
            
            with col_s4:
                st.markdown(
                    create_metric_card(
                        f"{stats['avg_sentence_length']:.1f}",
                        "Avg. Sentence Length",
                        "#43e97b",
                        "#38f9d7"
                    ),
                    unsafe_allow_html=True
                )
            
            # Vocabulary diversity chart
            st.markdown("---")
            diversity_ratio = stats['num_unique_words'] / stats['num_words'] if stats['num_words'] > 0 else 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=diversity_ratio * 100,
                title={'text': "Vocabulary Diversity"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FFE5E5"},
                        {'range': [30, 60], 'color': "#FFF4E5"},
                        {'range': [60, 100], 'color': "#E5F5FF"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Reference summary evaluation
            st.markdown("---")
            st.markdown("### 📈 Evaluation with Reference Summary")
            
            reference_text = st.text_area(
                "Enter reference summary for evaluation (optional)",
                height=150,
                help="Provide a human-written summary to calculate ROUGE scores"
            )
            
            if reference_text and 'current_summary' in st.session_state:
                if st.button("Calculate ROUGE Scores"):
                    with st.spinner('Calculating ROUGE scores...'):
                        evaluation = st.session_state.system.evaluate(
                            text,
                            st.session_state.current_summary['summary'],
                            reference_text
                        )
                        
                        if 'rouge_scores' in evaluation:
                            st.plotly_chart(
                                plot_rouge_scores(evaluation['rouge_scores']),
                                use_container_width=True
                            )
        else:
            st.info("📝 Generate a summary first to see analysis")
    
    with tab3:
        st.markdown("### 📚 Summarization History")
        
        if st.session_state.history:
            # Create DataFrame
            df = pd.DataFrame(st.session_state.history)
            
            # Display metrics
            col_h1, col_h2, col_h3 = st.columns(3)
            
            with col_h1:
                st.markdown(
                    create_metric_card(
                        len(st.session_state.history),
                        "Total Summaries",
                        "#667eea",
                        "#764ba2"
                    ),
                    unsafe_allow_html=True
                )
            
            with col_h2:
                avg_length = df['text_length'].mean()
                st.markdown(
                    create_metric_card(
                        f"{avg_length:.0f}",
                        "Avg. Text Length",
                        "#f093fb",
                        "#f5576c"
                    ),
                    unsafe_allow_html=True
                )
            
            with col_h3:
                most_used = df['method'].mode()[0] if not df.empty else "N/A"
                st.markdown(
                    create_metric_card(
                        most_used,
                        "Most Used Method",
                        "#4facfe",
                        "#00f2fe"
                    ),
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Display history table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("📋 No summarization history yet. Start summarizing texts to see your history here!")

if __name__ == "__main__":
    main()