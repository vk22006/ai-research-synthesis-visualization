# AI 研究合成与知识图谱构建器

[English](README.md) | [தமிழ்](README_TA.md) | 中文 | [हिन्दी](README_HI.md) | [Bahasa Indonesia](README_ID.md)

![应用程序主可视化](assets/app_visualization.gif)

一个用于加速 AI 研究的端到端工具，可自动获取论文、合成其内容，并将其组织为交互式知识图谱。

该应用使用自然语言处理 (NLP) 模型从 arXiv 论文中提取摘要和核心观点，并根据语义相似度可视化研究论文之间的联系。

## 功能特点

- **Automated Research Retrieval**: 根据您的搜索主题，直接从 arXiv 获取最新研究论文。
- **AI-Powered Synthesis**: 使用 Hugging Face Transformers 自动总结论文摘要并提取核心观点/贡献。
- **Semantic Similarity Analysis**: 使用 Sentence-Transformers 计算论文之间的语义相似度以发现联系。
- **Interactive Knowledge Graphs**: 使用 NetworkX 和 Pyvis 构建并渲染交互式知识图谱，展示不同研究论文之间的关联。
- **Modern Dashboard**: 直观的 Streamlit 前端，支持无缝交互、搜索配置和可视化探索。
- **Robust Backend API**: 基于 FastAPI 的后端架构，处理从数据检索到图谱生成的整个流程。

有关最新更新，请参阅 [更新日志 (UPDATE_LOG)](UPDATE_LOG.md)。

## 项目结构

```
.
├── app.py                      # FastAPI 后端应用
├── requirements.txt            # Python 依赖项
├── backend/                    # 后端核心逻辑
│   ├── fetch_papers.py         # arXiv 数据检索
│   ├── summarize.py            # 摘要生成
│   ├── claim_extractor.py      # 核心观点提取
│   ├── embeddings.py           # 相似度矩阵计算
│   ├── graph_builder.py        # 知识图谱生成
│   └── graph_visualizer.py     # 图谱 HTML 可视化
├── frontend/                   # 前端 UI
│   └── streamlit_app.py        # Streamlit 仪表盘应用
├── lib/                        # 附加工具/模块
└── data/                       # 生成输出的目录 (例如 graph.html)
```

## 技术栈

- **后端框架**: [FastAPI](https://fastapi.tiangolo.com/)
- **前端 UI**: [Streamlit](https://streamlit.io/)
- **NLP & 嵌入**: [Transformers](https://huggingface.co/docs/transformers/index), [Sentence-Transformers](https://sbert.net/), [PyTorch](https://pytorch.org/)
- **图谱与可视化**: [NetworkX](https://networkx.org/), [Pyvis](https://pyvis.readthedocs.io/)
- **数据处理**: [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)

## 快速开始

### 环境要求

确保已安装 Python 3.8+。建议使用虚拟环境。

### 安装步骤

1. 克隆此仓库或打开项目目录。
2. 安装所需的依赖项：

```bash
pip install -r requirements.txt
```

### 运行应用程序

该应用程序由后端 API 和前端仪表盘组成。您需要同时运行两者。

#### 1. 启动后端 (FastAPI)

在根目录下使用 `uvicorn` 运行 FastAPI 服务器：

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

后端 API 将在 `http://localhost:8000` 可用。您可以在 `http://localhost:8000/docs` 查看 API 文档。

#### 2. 启动前端 (Streamlit)

在新终端窗口中运行 Streamlit 应用：

```bash
streamlit run frontend/streamlit_app.py
```

前端仪表盘将自动在默认浏览器中打开 `http://localhost:8501`。

## 使用说明

1. 打开 Streamlit 前端。
2. 在侧边栏中输入 **研究主题 (Research Topic)**（例如 "Large Language Models", "Quantum Machine Learning", "Retrieval-Augmented Generation"）。
3. 调整 **最大结果数 (Max Results)**（获取的论文数量）和 **相似度阈值 (Similarity Threshold)**（在图谱中形成连接的最小相似度分数）。
4. 点击 **运行分析 (Run Analysis)**。
5. 系统将处理论文并显示提取的论文、摘要、核心观点以及交互式知识图谱可视化。

## 许可证

本项目基于 MIT 许可证开源，详情请参阅 [LICENSE](LICENSE)。
