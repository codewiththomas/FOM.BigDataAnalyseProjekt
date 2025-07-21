"""
Streamlit web interface for the Simple Research RAG system.
"""

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import json
import time
from pathlib import Path
import sys
import os

# Load environment variables
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.main import SimpleRAG
    from src.generator import AVAILABLE_MODELS
except ImportError:
    st.error("Could not import RAG modules. Make sure all dependencies are installed.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Simple Research RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        with st.spinner("Initializing RAG system..."):
            rag = SimpleRAG()
            st.session_state.rag_system = rag
        st.success("RAG system initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return False


def setup_data():
    """Setup data and build index."""
    if st.session_state.rag_system is None:
        st.error("RAG system not initialized")
        return False

    try:
        with st.spinner("Setting up data and building index..."):
            st.session_state.rag_system.setup_data()
        st.success("Data setup completed!")
        return True
    except Exception as e:
        st.error(f"Error setting up data: {e}")
        return False


def main():
    """Main Streamlit application."""
    st.title("🔍 Simple Research RAG")
    st.markdown("*Evaluating Language Models for German Document Retrieval*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Initialize system
        if st.button("Initialize RAG System", type="primary"):
            initialize_rag_system()

        # Model selection
        st.subheader("🤖 Model Selection")

        if st.session_state.rag_system:
            current_model = st.session_state.rag_system.config.get("model", {}).get("name", "unknown")
            st.info(f"Current model: **{current_model}**")

        model_options = list(AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            help="Choose the language model for generation"
        )

        if st.button("Switch Model"):
            if st.session_state.rag_system:
                with st.spinner(f"Switching to {selected_model}..."):
                    st.session_state.rag_system.switch_model(selected_model)
                st.success(f"Switched to {selected_model}")
                st.rerun()

        # Data setup
        st.subheader("📄 Data Setup")

        if st.button("Setup Data & Index"):
            setup_data()

        if st.button("Rebuild Index", help="Force rebuild the search index"):
            if st.session_state.rag_system:
                with st.spinner("Rebuilding index..."):
                    st.session_state.rag_system.setup_data(force_rebuild=True)
                st.success("Index rebuilt!")

        # System info
        st.subheader("ℹ️ System Info")
        if st.session_state.rag_system and st.session_state.rag_system.retriever.chunks:
            stats = st.session_state.rag_system.retriever.get_stats()
            st.metric("Documents Indexed", stats.get("num_chunks", 0))
            st.metric("Embedding Model", stats.get("embedding_model", "N/A"))

    # Main content area
    if st.session_state.rag_system is None:
        st.warning("Please initialize the RAG system using the sidebar.")
        st.markdown("""
        ### Getting Started
        1. Click **Initialize RAG System** in the sidebar
        2. Setup your data and build the search index
        3. Start asking questions or run evaluations
        """)
        return

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "📊 Evaluation", "📈 Results Analysis", "⚙️ Settings"])

    with tab1:
        query_interface()

    with tab2:
        evaluation_interface()

    with tab3:
        results_analysis()

    with tab4:
        settings_interface()


def query_interface():
    """Query interface tab."""
    st.header("💬 Ask Questions")

    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="Was ist die DSGVO?",
        help="Ask questions about your documents"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        top_k = st.slider("Number of contexts", 1, 10, 5)

    with col2:
        if st.button("🔍 Ask", type="primary", use_container_width=True):
            if query:
                process_query(query, top_k)
            else:
                st.warning("Please enter a question.")

    # Display query history
    if st.session_state.query_history:
        st.subheader("📜 Query History")

        for i, hist_item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {hist_item['question'][:80]}..."):
                st.markdown(f"**Answer:** {hist_item['answer']}")
                st.markdown(f"**Model:** {hist_item['model']}")
                st.markdown(f"**Time:** {hist_item['execution_time']:.2f}s")

                if hist_item['contexts']:
                    st.markdown("**Retrieved Contexts:**")
                    for j, context in enumerate(hist_item['contexts'][:3]):
                        st.markdown(f"{j+1}. {context[:200]}...")


def process_query(query: str, top_k: int):
    """Process a user query."""
    try:
        with st.spinner("Processing query..."):
            result = st.session_state.rag_system.query(query, top_k=top_k)

        # Display answer
        st.subheader("📝 Answer")
        st.markdown(result["response"])

        # Display metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Execution Time", f"{result.get('execution_time', 0):.2f}s")

        with col2:
            st.metric("Model", result.get("model", "Unknown"))

        with col3:
            cost = result.get("cost_estimate", 0)
            st.metric("Est. Cost", f"${cost:.4f}" if cost > 0 else "Free")

        # Display retrieved contexts
        if result.get("retrieved_contexts"):
            st.subheader("🔍 Retrieved Contexts")

            contexts = result["retrieved_contexts"]
            scores = result.get("retrieval_scores", [0] * len(contexts))

            for i, (context, score) in enumerate(zip(contexts, scores)):
                with st.expander(f"Context {i+1} (Score: {score:.3f})"):
                    st.markdown(context)

        # Add to history
        st.session_state.query_history.append({
            "question": query,
            "answer": result["response"],
            "model": result.get("model", "Unknown"),
            "execution_time": result.get("execution_time", 0),
            "contexts": result.get("retrieved_contexts", [])
        })

    except Exception as e:
        st.error(f"Error processing query: {e}")


def evaluation_interface():
    """Evaluation interface tab."""
    st.header("📊 Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.number_input(
            "Number of test questions",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of questions to evaluate"
        )

    with col2:
        st.markdown("### Available Metrics")
        st.markdown("""
        - **RAGAS**: Answer relevancy, faithfulness, context relevancy
        - **BERTScore**: Semantic similarity with reference answers
        - **Performance**: Execution time, tokens/second, cost
        """)

    if st.button("🚀 Run Evaluation", type="primary"):
        run_evaluation(test_size)

    # Display previous evaluation results
    if st.session_state.evaluation_results:
        st.subheader("📈 Latest Evaluation Results")

        results = st.session_state.evaluation_results

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        aggregates = results.get("aggregates", {})

        with col1:
            ragas_score = aggregates.get("ragas_answer_relevancy_mean", 0)
            if ragas_score > 0:
                st.metric("RAGAS Relevancy", f"{ragas_score:.3f}")
            else:
                st.metric("RAGAS Relevancy", "N/A")

        with col2:
            bertscore = aggregates.get("bertscore_f1_mean", 0)
            if bertscore > 0:
                st.metric("BERTScore F1", f"{bertscore:.3f}")
            else:
                st.metric("BERTScore F1", "N/A")

        with col3:
            avg_time = aggregates.get("execution_time_mean", 0)
            st.metric("Avg Time", f"{avg_time:.2f}s")

        with col4:
            total_cost = aggregates.get("cost_estimate_total", 0)
            st.metric("Total Cost", f"${total_cost:.4f}")

        # Detailed results table
        if st.checkbox("Show detailed results"):
            df = pd.DataFrame(results["results"])

            # Select available columns for display
            available_columns = ["question", "generated_answer"]

            # Add evaluation columns if they exist
            for col in ["ragas_answer_relevancy", "ragas_faithfulness", "bertscore_f1", "bertscore_precision"]:
                if col in df.columns:
                    available_columns.append(col)

            # Show only available columns
            if len(available_columns) > 2:
                st.dataframe(df[available_columns])
            else:
                st.dataframe(df[["question", "generated_answer"]])


def run_evaluation(test_size: int):
    """Run model evaluation."""
    try:
        with st.spinner(f"Running evaluation on {test_size} questions..."):
            results = st.session_state.rag_system.evaluate(test_size=test_size)

        st.session_state.evaluation_results = results
        st.success(f"Evaluation completed! Results saved.")

        # Show summary
        st.markdown("### Evaluation Summary")
        st.markdown(f"- **Model**: {results['model']}")
        st.markdown(f"- **Questions evaluated**: {results['test_size']}")
        st.markdown(f"- **Results saved to**: {results['saved_files']['aggregates']}")

        st.rerun()

    except Exception as e:
        st.error(f"Error running evaluation: {e}")


def results_analysis():
    """Results analysis tab."""
    st.header("📈 Results Analysis")

    # Load and compare model results
    results_dir = Path("results")

    if not results_dir.exists():
        st.warning("No evaluation results found. Run some evaluations first.")
        return

    # Model comparison
    try:
        comparison_df = st.session_state.rag_system.evaluator.compare_models("results")

        if not comparison_df.empty:
            st.subheader("🏆 Model Comparison")

            # Display comparison table
            st.dataframe(comparison_df, use_container_width=True)

            # Create visualizations
            create_comparison_plots(comparison_df)
        else:
            st.info("No comparison data available. Run evaluations for multiple models.")

    except Exception as e:
        st.error(f"Error loading comparison data: {e}")


def create_comparison_plots(df: pd.DataFrame):
    """Create comparison plots."""
    if df.empty:
        return

    # Performance comparison
    st.subheader("📊 Performance Comparison")

    # Metrics to plot
    metrics_to_plot = [
        ("ragas_answer_relevancy_mean", "RAGAS Answer Relevancy"),
        ("bertscore_f1_mean", "BERTScore F1"),
        ("execution_time_mean", "Execution Time (s)"),
        ("cost_estimate_total", "Total Cost ($)")
    ]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric[1] for metric in metrics_to_plot],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    for i, (metric_col, metric_name) in enumerate(metrics_to_plot):
        if metric_col in df.columns:
            row = (i // 2) + 1
            col = (i % 2) + 1

            fig.add_trace(
                go.Bar(
                    x=df["model"],
                    y=df[metric_col],
                    name=metric_name,
                    showlegend=False
                ),
                row=row, col=col
            )

    fig.update_layout(height=600, title_text="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)


def settings_interface():
    """Settings interface tab."""
    st.header("⚙️ Settings")

    # Configuration display
    st.subheader("📋 Current Configuration")

    if st.session_state.rag_system:
        config = st.session_state.rag_system.config
        st.json(config)

    # Model information
    st.subheader("🤖 Available Models")

    for model_id, model_info in AVAILABLE_MODELS.items():
        with st.expander(f"{model_id} - {model_info['name']}"):
            st.markdown(f"**Provider**: {model_info['provider']}")
            st.markdown(f"**Description**: {model_info['description']}")

    # System diagnostics
    st.subheader("🔧 System Diagnostics")

    if st.button("Run Diagnostics"):
        run_diagnostics()


def run_diagnostics():
    """Run system diagnostics."""
    st.markdown("### 🔍 System Diagnostics")

    # Check API keys
    st.markdown("#### API Keys")
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    st.markdown(f"- OpenAI API Key: {'✅ Set' if openai_key else '❌ Not set'}")
    st.markdown(f"- Groq API Key: {'✅ Set' if groq_key else '❌ Not set'}")

    # Check data directories
    st.markdown("#### Data Directories")
    data_dirs = ["data/documents", "data/faiss_index", "results"]

    for dir_path in data_dirs:
        exists = Path(dir_path).exists()
        st.markdown(f"- {dir_path}: {'✅ Exists' if exists else '❌ Missing'}")

    # Check dependencies
    st.markdown("#### Dependencies")
    dependencies = ["openai", "faiss", "sentence_transformers", "ragas", "bert_score"]

    for dep in dependencies:
        try:
            __import__(dep)
            st.markdown(f"- {dep}: ✅ Installed")
        except ImportError:
            st.markdown(f"- {dep}: ❌ Not installed")


if __name__ == "__main__":
    main()