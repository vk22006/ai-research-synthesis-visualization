# AI अनुसंधान संश्लेषण और ज्ञान ग्राफ निर्माता (AI Research Synthesis & Knowledge Graph Builder)

[English](README.md) | [தமிழ்](README_TA.md) | [中文](README_ZH.md) | हिन्दी | [Bahasa Indonesia](README_ID.md)

![एप्लिकेशन का मुख्य विज़ुअलाइज़ेशन](assets/app_visualization.gif)

AI अनुसंधान को गति देने के लिए एक एंड-टू-एंड टूल जो शोध पत्रों को स्वचालित रूप से प्राप्त करता है, उनकी सामग्री का संश्लेषण करता है, और उन्हें इंटरैक्टिव ज्ञान ग्राफ (Knowledge Graphs) में व्यवस्थित करता है।

यह एप्लिकेशन arXiv शोध पत्रों से सारांश और मुख्य दावों को निकालने के लिए नेचुरल लैंग्वेज प्रोसेसिंग (NLP) मॉडल का उपयोग करता है और शब्दार्थ समानता (semantic similarity) के आधार पर शोध पत्रों के बीच संबंधों को विज़ुअलाइज़ करता है।

## विशेषताएं

- **Automated Research Retrieval**: आपके खोज विषय के आधार पर सीधे arXiv से हाल के शोध पत्र प्राप्त करें।
- **AI-Powered Synthesis**: Hugging Face Transformers का उपयोग करके पेपर के सार को स्वचालित रूप से सारांशित करता है और मुख्य योगदानों को निकालता है।
- **Semantic Similarity Analysis**: संबंधों की खोज के लिए Sentence-Transformers का उपयोग करके शोध पत्रों के बीच शब्दार्थ समानता की गणना करता है।
- **Interactive Knowledge Graphs**: NetworkX और Pyvis का उपयोग करके एक इंटरैक्टिव ज्ञान ग्राफ बनाता है, जो दिखाता है कि विभिन्न शोध पत्र आपस में कैसे जुड़े हैं।
- **Modern Dashboard**: एक सहज Streamlit फ्रंटएंड जो सहज इंटरैक्शन, खोज कॉन्फ़िगरेशन और विज़ुअलाइज़ेशन अन्वेषण को सक्षम बनाता है।
- **Robust Backend API**: एक FastAPI-आधारित बैकएंड आर्किटेक्चर जो डेटा पुनर्प्राप्ति से लेकर ग्राफ निर्माण तक की पाइपलाइन को संभालता है।

हाल की अपडेट्स के लिए, [UPDATE LOG](UPDATE_LOG.md) देखें।

## प्रोजेक्ट संरचना

```
.
├── app.py                      # FastAPI बैकएंड एप्लिकेशन
├── requirements.txt            # Python डिपेंडेंसीज
├── backend/                    # बैकएंड कोर लॉजिक
│   ├── fetch_papers.py         # arXiv डेटा पुनर्प्राप्ति
│   ├── summarize.py            # सार का सारांशीकरण
│   ├── claim_extractor.py      # मुख्य दावा निष्कर्षण
│   ├── embeddings.py           # समानता मैट्रिक्स की गणना
│   ├── graph_builder.py        # ज्ञान ग्राफ जनरेशन
│   └── graph_visualizer.py     # ग्राफ HTML विज़ुअलाइज़ेशन
├── frontend/                   # फ्रंटएंड UI
│   └── streamlit_app.py        # Streamlit डैशबोर्ड एप्लिकेशन
├── lib/                        # अतिरिक्त उपयोगिताएँ/मॉड्यूल
└── data/                       # जनरेट किए गए आउटपुट के लिए निर्देशिका (उदा. graph.html)
```

## टेक स्टैक

- **बैकएंड फ्रेमवर्क**: [FastAPI](https://fastapi.tiangolo.com/)
- **फ्रंटएंड UI**: [Streamlit](https://streamlit.io/)
- **NLP और एंबेडिंग्स**: [Transformers](https://huggingface.co/docs/transformers/index), [Sentence-Transformers](https://sbert.net/), [PyTorch](https://pytorch.org/)
- **ग्राफ और विज़ुअलाइज़ेशन**: [NetworkX](https://networkx.org/), [Pyvis](https://pyvis.readthedocs.io/)
- **डेटा प्रोसेसिंग**: [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)

## शुरुआत कैसे करें

### पूर्वापेक्षाएँ (Prerequisites)

सुनिश्चित करें कि आपके पास Python 3.8+ इंस्टॉल है। वर्चुअल एनवायरनमेंट का उपयोग करने की सलाह दी जाती है।

### स्थापना (Installation)

1. इस रिपॉजिटरी को क्लोन करें या प्रोजेक्ट डायरेक्टरी खोलें।
2. आवश्यक डिपेंडेंसीज इंस्टॉल करें:

```bash
pip install -r requirements.txt
```

### एप्लिकेशन चलाना

एप्लिकेशन में एक बैकएंड API और एक फ्रंटएंड डैशबोर्ड शामिल है। आपको दोनों को एक साथ चलाना होगा।

#### 1. बैकएंड शुरू करें (FastAPI)

मूल निर्देशिका (root directory) से `uvicorn` का उपयोग करके FastAPI सर्वर चलाएं:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

बैकएंड API `http://localhost:8000` पर उपलब्ध होगा। आप `http://localhost:8000/docs` पर API दस्तावेज़ देख सकते हैं।

#### 2. फ्रंटएंड शुरू करें (Streamlit)

एक नई टर्मिनल विंडो में, Streamlit एप्लिकेशन चलाएं:

```bash
streamlit run frontend/streamlit_app.py
```

फ्रंटएंड डैशबोर्ड स्वचालित रूप से आपके डिफ़ॉल्ट ब्राउज़र में `http://localhost:8501` पर खुल जाएगा।

## उपयोग

1. Streamlit फ्रंटएंड खोलें।
2. साइडबार में, एक **अनुसंधान विषय (Research Topic)** दर्ज करें (उदा. "Large Language Models", "Quantum Machine Learning", "Retrieval-Augmented Generation")।
3. **अधिकतम परिणाम (Max Results)** (कितने पेपर प्राप्त करने हैं) और **समानता सीमा (Similarity Threshold)** (ग्राफ में कनेक्शन बनाने के लिए न्यूनतम समानता स्कोर) को समायोजित करें।
4. **विश्लेषण चलाएं (Run Analysis)** पर क्लिक करें।
5. सिस्टम पेपर्स को प्रोसेस करेगा और निकाले गए पेपर्स, उनके सारांश, मुख्य दावे और एक इंटरैक्टिव ज्ञान ग्राफ विज़ुअलाइज़ेशन प्रदर्शित करेगा।

## लाइसेंस

यह प्रोजेक्ट MIT लाइसेंस के तहत लाइसेंस प्राप्त है, अधिक जानकारी के लिए [LICENSE](LICENSE) देखें।
