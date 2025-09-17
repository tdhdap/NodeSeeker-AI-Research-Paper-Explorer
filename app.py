# app.py

# --- 1. Requirements ---
# Ensure you have a requirements.txt file with the following:
# streamlit
# langchain
# langchain-community
# langchain-google-genai
# langchain-experimental
# langchain-text-splitters
# pypdf
# pyvis
# networkx
# sentence-transformers

# --- 2. Imports ---
import streamlit as st
import os
import tempfile
from pyvis.network import Network
from langchain.prompts import PromptTemplate
import urllib.parse
import time  # Import the time module for delays
import random  # Import for jitter
from google.api_core.exceptions import ResourceExhausted  # Import for specific error handling

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.pydantic_v1 import BaseModel

# --- 3. App Configuration & API Key ---
st.set_page_config(
    page_title="NodeSeeker",
    page_icon="✏️️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Hahmlet:wght@100..900&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap');

    /* Global font setting */
    body * {
        font-family: 'Hahmlet', sans-serif !important;
    }

    /* Target the container for the tabs to center them */
    div[data-baseweb="tab-list"] {
        justify-content: center;
        border: none !important; /* Remove the border */
        box-shadow: none !important; /* Remove the shadow */
        padding: 0 !important; /* Remove padding */
    }

    /* A more specific selector to override Streamlit's default styles */
    div[data-baseweb="tab-list"] > button[data-testid="stTab"] {
        font-size: 22px !important;
        padding: 10px 25px !important;
        width: 350px !important;
        font-weight: bold !important;
        text-align: center !important;
        border: none !important; /* Remove individual button borders */
        box-shadow: none !important;
        background-color: transparent !important; /* Make button background transparent */
    }
</style>
""", unsafe_allow_html=True)

# Check if image exists, if not create placeholder
try:
    st.image("Add a heading (1).png", use_container_width=True)
except:
    # Fallback if image doesn't exist
    pass

st.markdown(
    "<h1 style='text-align: center; font-size: 3.5em;'>🕸 NodeSeeker.</h1>",
    unsafe_allow_html=True)
st.write("")
st.write("")

st.markdown("---")

# Securely get the API key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
except (KeyError, FileNotFoundError):
    with st.sidebar:
        st.header("API Key Configuration")
        api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.warning("Please enter your Gemini API Key to proceed.")
            st.stop()

# --- 4. Session State Initialization ---
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'summary_generated' not in st.session_state:
    st.session_state.summary_generated = False
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'node_ids' not in st.session_state:
    st.session_state.node_ids = []
if 'html_path' not in st.session_state:
    st.session_state.html_path = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'show_file_uploader' not in st.session_state:
    st.session_state.show_file_uploader = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""

# --- 5. Sidebar Content ---
with st.sidebar:
    st.divider()
    st.header("📖 How to Use")
    st.markdown("""
    <div style="font-size: 1.3em;">
    1.  📤 Upload a PDF of the research paper you want to analyze.
    <br><br>
    2.  ✨ Generate a summary or a knowledge graph.
    <br><br>
    3.  🔬 Interact with the graph and use Deep Dive to explore any concept.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Enhanced Upload Button Logic
    if st.button("Upload a File", use_container_width=True):
        if st.session_state.file_processed:
            # If file already processed, refresh the page AND show uploader
            st.session_state.file_processed = False
            st.session_state.graph_generated = False
            st.session_state.summary_generated = False
            st.session_state.all_chunks = []
            st.session_state.node_ids = []
            st.session_state.html_path = ""
            st.session_state.summary = ""
            st.session_state.uploaded_file = None
            st.session_state.show_file_uploader = True  # Show uploader immediately
            st.session_state.full_text = ""
            st.rerun()
        else:
            # Show file uploader
            st.session_state.show_file_uploader = True

    # File uploader widget (shown conditionally)
    if st.session_state.show_file_uploader and not st.session_state.file_processed:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            # Store the uploaded file in session state
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_processed = True
            st.session_state.show_file_uploader = False
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            st.rerun()

# --- 6. Core Logic: Graph & Text Processing ---

NARRATIVE_PROMPT = PromptTemplate.from_template(
    """
    You are a research analyst. Your task is to deconstruct the provided text from a scientific paper 
    into a knowledge graph that tells the story of the research.

    **Instructions:**
    1.  Identify the core entities in the text. Classify them into one of the following types: 
        - `Main_Objective`: The single, overarching goal or thesis of the paper.
        - `Problem`: The specific challenge or gap the research addresses.
        - `Method`: The techniques, algorithms, or experiments used.
        - `Finding`: A concrete result, conclusion, or key data point (including metrics).
        - `Concept`: An important theoretical idea or background term.

    2.  Establish connections between these entities using ONLY the following relationship types:
        - `ADDRESSES`: Connects a `Method` or `Main_Objective` to a `Problem`.
        - `USES`: Connects a `Method` to a `Concept` or another `Method`.
        - `PRODUCES`: Connects a `Method` to a `Finding`.
        - `MEASURES`: Connects a `Finding` (that is a metric) to the `Method` it evaluates.
        - `SUPPORTS`: Connects a `Finding` back to the `Main_Objective`.

    **Output Format:**
    - Provide your output as a list of JSON objects for nodes and relationships.

    **Example Node:**
    {{ "id": "Dual-Objective HR Analytics", "type": "Main_Objective" }}

    **Example Relationship:**
    {{ "source": "F1-Score", "target": "Logistic Regression", "type": "MEASURES" }}

    Text to analyze:
    {input}
    """
)


# --- 7. New Feature: Focus Mode & Summarization ---

@st.cache_data
def generate_node_deep_dive(node_id, context_str):
    """
    Generates a detailed explanation for a specific node using a context string.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.1)

    prompt = f"""
    You are a helpful research assistant. Your task is to provide a detailed explanation of a specific concept from a research paper.

    **Concept to Explain:** "{node_id}"

    **Context from the Paper:**
    ---
    {context_str}
    ---

    Based on the provided context, please generate the following using Markdown for formatting:

    ### Definition
    A concise, one-sentence definition of the concept.

    ### Deep Dive
    A more detailed explanation of how this concept is used or discussed specifically within this paper, drawing directly from the context.
    """

    response = llm.invoke(prompt)
    return response.content


@st.cache_data
def generate_structured_summary(full_text):
    """
    Generates a structured summary of the entire paper.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.3)

    prompt = f"""
    You are an expert academic tutor. Your task is to create a concise, structured summary of the following research paper text.
    The summary should be easy for a student to understand and follow the logical flow of the paper.

    **Instructions:**
    1. Read the entire text to understand its purpose, methods, and conclusions.
    2. Identify the main sections of the paper based on its content (e.g., Introduction, Methodology, Results, Conclusion).
    3. For each section, write a short, clear paragraph summarizing its key points.
    4. Use the paper's own headings where possible to structure your summary.
    5. **Use Markdown headings (e.g., '## Introduction' or '### 2.1 Attrition Classification') for all section titles.** Do not just bold them.
    **Full Text of the Paper:**
    ---
    {full_text}
    ---

    **Now, please generate the structured summary:**
    """
    response = llm.invoke(prompt)
    return response.content


# --- 8. Visualization Logic ---

def visualize_graph(graph_documents):
    """
    Visualizes the graph using a physics-based layout.
    """
    height = '652px'

    net = Network(
        notebook=False,
        cdn_resources='remote',
        height=height,
        width='100%',
        bgcolor='#FFFFFF',
        font_color='black',
        select_menu=False,
        filter_menu=False,
        directed=True
    )

    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 24,
          "face": "Helvetica",
          "color": "#000000"
        },
        "borderWidth": 2
      },
      "edges": {
        "font": {
          "size": 12,
          "align": "top"
        },
        "smooth": {
          "type": "continuous"
        },
        "color": {
          "inherit": true
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.5 }
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -300,
          "centralGravity": 0.009,
          "springLength": 175,
          "springConstant": 0.09
        },
        "minVelocity": 0.6,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    nodes_dict = {}
    all_node_ids = set()

    for doc in graph_documents:
        for node in doc.nodes:
            nodes_dict[node.id] = node
            all_node_ids.add(node.id)
        for rel in doc.relationships:
            if rel.source.id not in nodes_dict:
                nodes_dict[rel.source.id] = rel.source
                all_node_ids.add(rel.source.id)
            if rel.target.id not in nodes_dict:
                nodes_dict[rel.target.id] = rel.target
                all_node_ids.add(rel.target.id)

    for node_id, node in nodes_dict.items():
        net.add_node(node.id, label=node.id, title=f"Type: {node.type}", color=get_node_color(node.type))

    for doc in graph_documents:
        for rel in doc.relationships:
            if rel.source.id in nodes_dict and rel.target.id in nodes_dict:
                net.add_edge(rel.source.id, rel.target.id, label=rel.type, title=rel.type)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name, sorted(list(all_node_ids))


def get_node_color(node_type):
    """Assigns a color to a node based on its type."""
    colors = {
        "Main_Objective": "#6A1B9A",  # Dark Purple
        "Problem": "#8E24AA",  # Medium Purple
        "Method": "#AB47BC",  # Lighter Purple
        "Finding": "#CE93D8",  # Lightest Purple
        "Concept": "#D1C4E9",  # Very Light Lavender
    }
    return colors.get(node_type, "#BDBDBD")  # Default to grey


# --- 9. Sample Data for Welcome Screen ---
class Node(BaseModel):
    id: str
    type: str


class Relationship(BaseModel):
    source: Node
    target: Node
    type: str


class GraphDocument(BaseModel):
    nodes: list[Node]
    relationships: list[Relationship]


@st.cache_data
def get_sample_graph():
    """Returns a pre-defined graph document for the welcome screen."""
    nodes = [
        Node(id="Predict Customer Churn", type="Main_Objective"),
        Node(id="High Customer Attrition", type="Problem"),
        Node(id="Logistic Regression", type="Method"),
        Node(id="95% Predictive Accuracy", type="Finding"),
        Node(id="Statistical Modeling", type="Concept"),
    ]

    relationships = [
        Relationship(source=nodes[0], target=nodes[1], type="ADDRESSES"),
        Relationship(source=nodes[2], target=nodes[0], type="APPLIES_TO"),
        Relationship(source=nodes[2], target=nodes[4], type="IS_BASED_ON"),
        Relationship(source=nodes[2], target=nodes[3], type="YIELDS"),
        Relationship(source=nodes[3], target=nodes[0], type="VALIDATES"),
    ]
    return [GraphDocument(nodes=nodes, relationships=relationships)]


# --- 10. Main Application Logic ---

# Welcome Screen Layout (only shows if no file is uploaded yet)
if not st.session_state.file_processed:
    with st.container():

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<h2 style='text-align: center;'>Example Summary</h3>", unsafe_allow_html=True)


            with st.container(border=True):
                sample_summary = """
                ## Introduction
                This research addresses the critical business challenge of customer churn prediction in subscription-based services. With customer acquisition costs continuously rising, retaining existing customers has become more cost-effective than acquiring new ones.

                ## Methodology
                The study employs machine learning techniques, specifically logistic regression models, to analyze customer behavior patterns and predict churn probability. The methodology incorporates statistical modeling principles to identify key factors influencing customer retention.

                ## Key Findings
                The implemented logistic regression model achieved a remarkable 95% predictive accuracy in identifying customers likely to churn. This high accuracy rate demonstrates the effectiveness of statistical modeling approaches in addressing customer attrition challenges.

                ## Conclusion
                The research successfully demonstrates that predictive analytics can significantly improve customer retention strategies, providing businesses with actionable insights to proactively address churn risks.
                """
                st.markdown(sample_summary)

        with col2:
            st.markdown("<h2 style='text-align: center;'>Example Knowledge Graph</h3>", unsafe_allow_html=True)
            st.info("Click and drag the nodes to explore the connections.")

            sample_graph_docs = get_sample_graph()
            html_path, _ = visualize_graph(sample_graph_docs)

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            st.components.v1.html(html_content, height=800)

# File Processing and Analysis (shows when file is uploaded)
elif st.session_state.file_processed and st.session_state.uploaded_file:

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            st.session_state.uploaded_file.seek(0)
            tmp_file.write(st.session_state.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
            st.error("Error: The uploaded file appears to be empty or corrupted. Please try uploading again.")
            st.session_state.file_processed = False
            st.session_state.uploaded_file = None
            st.rerun()

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])

        if not full_text.strip():
            st.error("Error: No text could be extracted from the PDF. Please ensure the file contains readable text.")
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            st.session_state.file_processed = False
            st.session_state.uploaded_file = None
            st.rerun()

        st.session_state.full_text = full_text
        splitter = SentenceTransformersTokenTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.all_chunks = splitter.split_text(full_text)

        try:
            os.unlink(tmp_file_path)
        except:
            pass

    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
        st.info("Please try uploading the file again or ensure the PDF is not corrupted.")
        st.session_state.file_processed = False
        st.session_state.uploaded_file = None
        st.rerun()

    st.markdown("""
        <style>
        button[data-baseweb="tab"] {
            font-size: 20px !important;
            padding: 10px 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    tab1, tab2, = st.tabs(["📄 Summary", "🕸️ Knowledge Graph"])

    with tab1:
        with st.container(border=True, height=800):
            st.markdown("<h2 style='text-align: center;'>📄 Summary</h3>", unsafe_allow_html=True)

            if st.button("Generate Summary", key="summary_btn"):
                if st.session_state.full_text:
                    with st.spinner("Generating comprehensive summary..."):
                        summary = generate_structured_summary(st.session_state.full_text)
                        st.session_state.summary = summary
                        st.session_state.summary_generated = True
                else:
                    st.error("No text available for summary generation.")

            if st.session_state.summary_generated and st.session_state.summary:
                st.markdown(st.session_state.summary)

    with tab2:
        with st.container():
            column1, column2 = st.columns([1, 1])
            with column1:
                st.markdown("### 🧠 Knowledge Graph")

                chunk_limit = int(st.slider(
                    "Adjust Complexity (Lower settings are faster)",
                    min_value=float(1),
                    max_value=float(len(st.session_state.all_chunks)),
                    value=float(min(10, len(st.session_state.all_chunks))),
                    step=0.1
                ))

                if st.button("Generate Knowledge Graph", key="graph_btn"):
                    if st.session_state.all_chunks:
                        with st.spinner("Generating graph... This may take a moment."):
                            try:
                                # FIX: Process chunks as a single batch, not one-by-one
                                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.3)
                                llm_transformer = LLMGraphTransformer(llm=llm, prompt=NARRATIVE_PROMPT,
                                                                      strict_mode=False)

                                # Select the chunks to process based on the slider
                                chunks_to_process = st.session_state.all_chunks[:chunk_limit]
                                documents = [Document(page_content=chunk) for chunk in chunks_to_process]

                                # Make a single, efficient API call
                                graph_documents = llm_transformer.convert_to_graph_documents(documents)

                                if graph_documents:
                                    st.session_state.graph_documents = graph_documents
                                    html_path, node_ids = visualize_graph(graph_documents)
                                    st.session_state.html_path = html_path
                                    st.session_state.node_ids = node_ids
                                    st.session_state.graph_generated = True
                                    st.success("✅ Knowledge graph generated!")
                                else:
                                    st.warning(
                                        "Could not extract any graph data from the selected text. Try adjusting the text or increasing the graph detail.")

                            except Exception as e:
                                st.error(f"An error occurred during graph generation: {e}")

                if st.session_state.graph_generated and st.session_state.html_path:
                    with open(st.session_state.html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)

            with column2:
                st.markdown("### 🔍 Deep Dive")

                if st.session_state.graph_generated and st.session_state.node_ids:
                    selected_node = st.selectbox(
                        "",
                        st.session_state.node_ids,
                        key="focus_node_selector",
                        index=None,
                        placeholder="Select a concept to explore"
                    )

                    if st.button("Generate Deep Dive", key="focus_btn"):
                        if selected_node:
                            with st.spinner(f"Generating deep dive for {selected_node}..."):
                                context_str = "\n\n".join(st.session_state.all_chunks)
                                deep_dive_content = generate_node_deep_dive(selected_node, context_str)
                                st.markdown(f"## {selected_node}")
                                st.markdown(deep_dive_content)
                else:
                    st.info("Generate a knowledge graph first to use Deep Dive.")






















