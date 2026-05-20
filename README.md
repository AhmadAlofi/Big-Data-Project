# Arabic AI-Text Detection System

A distributed system that detects whether Arabic text is written by a **Human** or **AI** (ChatGPT, GPT-4, Llama, etc.) using Apache Spark.

## Key Result

| Metric | Value |
|--------|-------|
| **Accuracy** | **95.69%** |
| Best Model | Linear SVM |
| Dataset Size | 41,940 Arabic abstracts |

## 🔧 How It Works

1. **Clean** Arabic text (remove diacritics, normalize letters)
2. **Extract** features (word frequency + text length + punctuation)
3. **Train** classification model
4. **Predict** in real-time

## Tech Stack

- Apache Spark (big data processing)
- Python
- Scikit-learn / MLlib
- PyArabic, NLTK

##  Performance

- Optimal speed: **4 partitions** (0.14 seconds)
- Real-time streaming capable

##  Achievements

 41,940 Arabic texts processed  
 95.69% detection accuracy  
 Real-time inference  
 Scalable architecture

##  Quick Start

```bash
git clone https://github.com/yourusername/arabic-ai-detection-spark.git
cd arabic-ai-detection-spark
pip install -r requirements.txt
jupyter notebook main_detection_pipeline.ipynb