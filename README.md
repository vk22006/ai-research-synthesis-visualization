# AI Research Synthesis & Knowledge Graph Builder

![Main visualizations of the application](assets/app_visualization.gif)

An end-to-end tool to accelerate AI research by automatically fetching papers, synthesizing their contents, and organizing them into interactive knowledge graphs. 

This application uses Natural Language Processing (NLP) models to extract summaries and core claims from arXiv papers and visualizes the connections between research papers based on semantic similarity.

## Features

- **Automated Research Retrieval**: Fetch recent research papers directly from arXiv based on your search topic.
- **AI-Powered Synthesis**: Automatically summarizes paper abstracts and extracts the core claims/contributions using Hugging Face Transformers.
- **Semantic Similarity Analysis**: Computes the semantic similarity between papers using Sentence-Transformers to discover connections.
- **Interactive Knowledge Graphs**: Builds and renders an interactive knowledge graph using NetworkX and Pyvis, illustrating how different research papers are related.
- **Modern Dashboard**: An intuitive Streamlit frontend enabling seamless interaction, search configuration, and visualization exploration.
- **Robust API Backend**: A FastAPI-based backend architecture handling the pipeline from data retrieval to graph generation.

For recent updates, refer [UPDATE_LOG](UPDATE_LOG.md).

## Project Structure

```
.
├── app.py                      # FastAPI backend application
├── requirements.txt            # Python dependencies
├── backend/                    # Backend core logic
│   ├── fetch_papers.py         # arXiv data retrieval
│   ├── summarize.py            # Abstract summarization
│   ├── claim_extractor.py      # Core claim extraction
│   ├── embeddings.py           # Similarity matrix computation
│   ├── graph_builder.py        # Knowledge graph generation
│   └── graph_visualizer.py     # Graph HTML visualization
├── frontend/                   # Frontend UI
│   └── streamlit_app.py        # Streamlit dashboard application
├── lib/                        # Additional utilities/modules
└── data/                       # Directory for generated outputs (e.g., graph.html)
```

## Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **NLP & Embeddings**: [Transformers](https://huggingface.co/docs/transformers/index), [Sentence-Transformers](https://sbert.net/), [PyTorch](https://pytorch.org/)
- **Graph & Visualization**: [NetworkX](https://networkx.org/), [Pyvis](https://pyvis.readthedocs.io/)
- **Data Processing**: [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation

1. Clone this repository or open the project directory.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

The application consists of a backend API and a frontend dashboard. You need to run both concurrently.

#### 1. Start the Backend (FastAPI)

Run the FastAPI server using `uvicorn` (from the root directory):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
The backend API will be available at `http://localhost:8000`. You can view the API documentation at `http://localhost:8000/docs`.

#### 2. Start the Frontend (Streamlit)

In a new terminal window, run the Streamlit application:

```bash
streamlit run frontend/streamlit_app.py
```
The frontend dashboard will automatically open in your default browser at `http://localhost:8501`.

## Usage

1. Open the Streamlit frontend.
2. In the sidebar, enter a **Research Topic** (e.g., "Large Language Models", "Quantum Machine Learning", "Retrieval-Augmented Generation").
3. Adjust the **Max Results** (how many papers to fetch) and the **Similarity Threshold** (minimum similarity score to form a connection in the graph).
4. Click **Run Analysis**. 
5. The system will process the papers and display the Extracted Papers, their Summaries, extracted Claims, and an Interactive Knowledge Graph visualization.

## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) for more details.
