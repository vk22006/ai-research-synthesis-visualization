# AI ஆராய்ச்சி தொகுப்பு மற்றும் அறிவு வரைபடம் உருவாக்குதல்

[English](README.md) | தமிழ் | [中文](README_ZH.md) | [हिन्दी](README_HI.md) | [Bahasa Indonesia](README_ID.md)

![பயன்பாட்டின் செயல்பாடு](assets/app_visualization.gif)

AI ஆராய்ச்சியை தானாகவே சுருக்கம் செய்து, அறிவு வரைபடம் மூலம் காட்சிப்படுத்தி, கட்டுரைகளை ஆராய உதவும் ஒரு கருவி.

இது NLP மாடல்களைப் பயன்படுத்தி arXiv கட்டுரைகளை தானாகவே சுருக்கி, அவற்றின் தொடர்புகளைக் கண்டறிந்து, அறிவு வரைபடத்தை உருவாக்குகிறது.

## சிறப்பம்சங்கள்

- **Automated Research Retrieval**: உங்கள் தேடல் தலைப்புக்கு ஏற்ப arXiv இலிருந்து சமீபத்திய ஆராய்ச்சி கட்டுரைகளை தானாகப் பெறுகிறது.
- **AI-Powered Summarization**: Hugging Face Transformers ஐப் பயன்படுத்தி தானாகவே கட்டுரைகளின் சுருக்கங்களையும் முக்கியக் கோரிக்கைகளையும் பிரித்தெடுக்கிறது.
- **Semantic Similarity Analysis**: தொடர்பு கண்டறியण्यासाठी கட்டுரைகளுக்கு இடையேயான சொற்பொருள் ஒற்றுமையை கணக்கிடுகிறது.
- **Interactive Knowledge Graphs**: ஆராய்ச்சியாளர்கள் எவ்வாறு வேறுபடுகிறார்கள் என்பதை விளக்கும் ஒரு ஊடாடும் அறிவு வரைபடத்தை உருவாக்குகிறது.
- **Modern Dashboard**: தடையற்ற ஊடாடலுக்கும் காட்சிப்படுத்தல் ஆராய்வதற்கும் உதவும் ஒரு பயனர் நட்பு Streamlit முன்முனையை வழங்குகிறது.
- **Robust API Backend**: தரவைப் பெறுவதில் இருந்து வரைபடம் வரை ஆற்றலை ஆற்றல் அளிக்கும் FastAPI அடிப்படையிலான கட்டிடக்கலை.

சமீப புதுப்பிப்புகளுக்கு, [UPDATE LOG](UPDATE_LOG.md) இல் காண்க.

## திட்ட கட்டமைப்பு

```
.
├── app.py                      # FastAPI backend application
├── requirements.txt            # Python dependencies
├── backend/                    # Backend core logic
│   ├── fetch_papers.py         # arXiv தரவுகளை பெறுதல்
│   ├── summarize.py            # கட்டுரைகளின் சுருக்கம்
│   ├── claim_extractor.py      # முக்கிய கோரிக்கைகளின் தொகுப்பு
│   ├── embeddings.py           # தொடர்பு matrix கணக்கிடுதல்
│   ├── graph_builder.py        # அறிவு வரைபடம் உருவாக்குதல்
│   └── graph_visualizer.py     # வரைபடம் HTML காட்சிப்படுத்தல்
├── frontend/                   # Frontend UI
│   └── streamlit_app.py        # Streamlit டாஷ்போர்டு 
├── lib/                        # கூடுதல் பயன்பாடுகளின் கோப்புகள்
└── data/                       # உருவாக்கப்பட்ட தரவுகளை சேமிக்கும் இடம் 
```

## தொழில்நுட்ப அமைப்பு

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **NLP & Embeddings**: [Transformers](https://huggingface.co/docs/transformers/index), [Sentence-Transformers](https://sbert.net/), [PyTorch](https://pytorch.org/)
- **Graph & Visualization**: [NetworkX](https://networkx.org/), [Pyvis](https://pyvis.readthedocs.io/)
- **Data Processing**: [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)

## தொடங்குவது

### முன் தேவைகள் (Prerequisites)

Python 3.8+ நிறுவப்பட்டுள்ளதா என்பதை உறுதிசெய்யவும். ஒரு விர்ச்சுவல் என்விரோன்மென்டை பயன்படுத்துவது பரிந்துரைக்கப்படுகிறது.

### நிறுவுதல்

1. இந்த ரெபோசிட்டரியை கிளிக் செய்யவும் அல்லது திட்ட கோப்புறைக்கு செல்லவும்.
2. தேவையான பைதான் டிபெண்டன்ஸிகளை install செய்யவும்:

```bash
pip install -r requirements.txt
```

### அப்ளிகேஷனை இயக்குதல்

இந்த அப்ளிகேஷன் backend API மற்றும் frontend dashboard கொண்டது. இரண்டையும் ஒன்று பின் ஒன்றாக இயக்க வேண்டும்.

#### 1. Start the Backend (FastAPI)

Run the FastAPI server using `uvicorn` (மூல அடைவில்):

```bash
uvicorn app:app --reload --host [IP_ADDRESS] --port 8000
```

The backend API will be available at `http://localhost:8000`. You can view the API documentation at `http://localhost:8000/docs`.

#### 2. Start the Frontend (Streamlit)

புதிய terminal window-வில் :

```bash
streamlit run frontend/streamlit_app.py
```

மேலகன்பதை செய்தால் நம்மாளுடைய டாஷ்போர்டு தானாகவே நம்மாளுடைய உலாவியில் `http://localhost:8501` இல் தோன்றும்.
