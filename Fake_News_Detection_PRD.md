# Product Requirements Document (PRD)
## Fake News Detection using Machine Learning / NLP

---

### 1. Project Overview

**Project Name:** TruthLens – Fake News Detection System

**One-line description:** A machine learning-based web application that classifies news articles/headlines as "Real" or "Fake" using Natural Language Processing (NLP), with a confidence score and explainability layer.

**Problem Statement:** The rapid spread of misinformation on social media and digital platforms has made it increasingly difficult for readers to distinguish credible news from fabricated content. This project aims to build an ML/NLP-based system that automatically analyzes text content and predicts its authenticity, helping users make informed decisions.

**Target Users:**
- Students/researchers studying misinformation
- General public verifying news before sharing
- Content moderators (as an assistive tool)
- Recruiters/evaluators reviewing this as an academic/portfolio project

---

### 2. Goals & Objectives

| Goal | Success Metric |
|---|---|
| Build a working classifier for fake vs real news | Accuracy ≥ 90% on test set |
| Provide an intuitive UI for real-time prediction | User can get a prediction in < 3 seconds |
| Make the model interpretable | Show top contributing words/phrases per prediction |
| Deployable, demo-ready project | Live hosted link + GitHub repo with README |

**Out of Scope (v1):**
- Real-time web scraping/crawling of live news sites
- Multi-language support (English only for v1)
- Video/image-based misinformation detection
- Browser extension (can be a stretch goal / v2)

---

### 3. Dataset

**Recommended datasets:**
- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Fake.csv + True.csv, ~44,000 articles)
- LIAR Dataset (Politifact-based, short statements with 6-way labels)
- ISOT Fake News Dataset

**Data fields typically used:** `title`, `text`, `subject`, `date`, `label (0 = fake, 1 = real)`

**Preprocessing steps:**
1. Lowercasing, punctuation removal, stopword removal
2. Tokenization & lemmatization (NLTK/spaCy)
3. Handling missing values and duplicates
4. Train/validation/test split (e.g., 70/15/15)
5. Feature extraction: TF-IDF (baseline) → later Word2Vec/GloVe/BERT embeddings

---

### 4. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | User can input a news headline/article text into a text box | Must-have |
| FR2 | System returns a prediction: Real / Fake | Must-have |
| FR3 | System shows a confidence score (%) | Must-have |
| FR4 | System highlights key words influencing the decision (explainability via LIME/SHAP) | Should-have |
| FR5 | User can view model performance metrics (accuracy, precision, recall, F1, confusion matrix) on a dedicated page | Should-have |
| FR6 | User can paste a URL and system extracts article text automatically (via newspaper3k) | Could-have (stretch) |
| FR7 | Prediction history log (session-based, no login needed for v1) | Could-have |
| FR8 | API endpoint for programmatic access | Should-have |

---

### 5. Non-Functional Requirements

- **Performance:** Inference latency < 2 seconds per request
- **Scalability:** Should handle at least 50 concurrent requests (for demo/portfolio purposes)
- **Usability:** Clean, minimal, mobile-responsive UI
- **Reliability:** Graceful error handling for empty input, non-English text, very short text
- **Maintainability:** Modular code structure separating data processing, model training, and serving layers

---

### 6. Tech Stack (Suggested)

| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy, NLTK/spaCy |
| ML/Modeling | Scikit-learn (Logistic Regression, Naive Bayes, SVM, Random Forest for baselines); TensorFlow/PyTorch + HuggingFace Transformers (BERT/DistilBERT) for advanced model |
| Explainability | LIME or SHAP |
| Backend/API | FastAPI or Flask |
| Frontend | React.js (aligns with your existing skillset) or simple HTML/CSS/JS |
| Database (optional) | MySQL/SQLite for storing prediction logs |
| Deployment | Render/Railway/HuggingFace Spaces (backend), Vercel/Netlify (frontend) |
| Version Control | Git/GitHub |

---

### 7. Model Development Plan

**Phase 1 – Baseline Model**
- TF-IDF vectorization + Logistic Regression / Naive Bayes / Linear SVM
- Establish baseline accuracy, precision, recall, F1

**Phase 2 – Improved Model**
- Try ensemble methods (Random Forest, XGBoost) on TF-IDF features
- Hyperparameter tuning via GridSearchCV

**Phase 3 – Deep Learning / Transformer Model**
- Fine-tune DistilBERT or BERT-base on the dataset (using HuggingFace)
- Compare performance against classical ML baselines

**Phase 4 – Explainability**
- Integrate LIME/SHAP to show which words pushed the prediction toward "Fake" or "Real"

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

---

### 8. System Architecture (High-Level)

```
User (Browser)
     |
     v
Frontend (React) --> REST API (FastAPI/Flask)
                            |
                            v
                    Preprocessing Pipeline
                            |
                            v
                    Trained ML/DL Model (.pkl / HuggingFace model)
                            |
                            v
                Prediction + Confidence + Explanation
                            |
                            v
                    Response back to Frontend
```

---

### 9. Project Milestones / Timeline (Suggested — 6 weeks)

| Week | Milestone |
|---|---|
| 1 | Dataset collection, EDA, preprocessing pipeline |
| 2 | Baseline model (TF-IDF + Logistic Regression/Naive Bayes) |
| 3 | Model improvement + evaluation, try ensemble models |
| 4 | Transformer-based model (DistilBERT fine-tuning) |
| 5 | Backend API + Frontend integration |
| 6 | Explainability (LIME/SHAP), deployment, README, demo polish |

---

### 10. Deliverables

- GitHub repository with clean, documented code
- Trained model artifact(s)
- Working deployed web app (frontend + backend)
- README with problem statement, architecture diagram, setup instructions, results
- (Optional) A short project report/PPT for academic submission

---

### 11. Risks & Considerations

- **Dataset bias:** Public fake news datasets can be topic/time-biased (e.g., politics-heavy); mention this as a limitation.
- **Overfitting:** Classical ML models can overfit to specific phrasing; validate on out-of-distribution samples if possible.
- **Ethical framing:** Present the tool as an "assistive credibility signal," not an absolute truth arbiter — avoid overclaiming accuracy in real-world deployment.

---

### 12. Future Enhancements (v2+)

- Browser extension for real-time flagging on social media/news sites
- Multi-language support
- Source credibility scoring (cross-referencing publisher reputation)
- Fine-tuning on more recent/diverse datasets to reduce topic bias
