# Projects

Projects are grouped by evidence source. The first two are the featured projects in the November 2026 CV. Other repositories and previously discussed projects should not be presented as professional client deployments unless explicitly confirmed.

## 0. Portfolio AI Assistant — RAG Chatbot for Recruiters

**Evidence:** Personal project, integrated into the portfolio website.

### Overview
A Retrieval-Augmented Generation (RAG) chatbot embedded in Seif's portfolio website. It answers recruiter questions about his education, skills, projects, and experience, grounded entirely in his real portfolio documents — no hallucinations.

### Technologies listed
Python, Sentence-Transformers (all-MiniLM-L6-v2), ChromaDB, LangChain, Groq API (Llama 3), FastAPI, HTML/CSS/JavaScript.

### Architecture and responsibilities described
- **Document pipeline:** Loads Markdown knowledge-base files and splits them into chunks by Markdown headers using LangChain.
- **Embedding and vector storage:** Generates 384-dimensional embeddings locally with Sentence-Transformers and stores them in ChromaDB for persistent semantic search.
- **RAG retrieval:** Retrieves the top-5 most relevant chunks for each question using cosine-similarity search.
- **LLM generation:** Sends the question plus retrieved context to Groq's Llama model with a strict system prompt that enforces grounding rules (no invented facts, no unsupported claims).
- **API and integration:** FastAPI REST endpoint (`POST /chat`) consumed by a JavaScript chat widget embedded in the portfolio front-end.

---

## 1. End-to-End MLOps Platform — Telecom Customer Churn Detection

**Evidence:** November 2026 CV.

### Overview
An end-to-end platform for telecom customer-churn prediction. The described workflow covers data ingestion, transformation and quality checks, data versioning, experiment tracking, orchestration, API deployment, CI/CD, and monitoring.

### Technologies listed
Python, dlt, dbt, DuckDB, DVC, MLflow, FastAPI, Docker, Dagster, GitHub Actions, Prometheus, Grafana.

### Architecture and responsibilities described
- **DataOps and quality:** Automated data ingestion and transformation using dlt, DuckDB, and dbt, with integrated data-quality tests.
- **Versioning and experiment tracking:** DVC for data versioning and MLflow for tracking model metrics/experiments.
- **Orchestration:** Dagster for coordinating the overall workflow.
- **Serving and operations:** Dockerized FastAPI REST API, CI/CD through GitHub Actions, and monitoring with Prometheus and Grafana.
- **Drift monitoring:** PSI (Population Stability Index) is mentioned as part of the monitoring approach.

### Not documented
The CV does not state the exact dataset size, model family, model scores, production traffic, or verified business impact. Do not invent these values.

---

## 2. Real-Time Banking Fraud Detection Platform

**Evidence:** November 2026 CV.

### Overview
A real-time fraud-detection platform using Kafka for ingestion and PySpark Structured Streaming for distributed micro-batch processing on YARN.

### Technologies listed
Python, PySpark, Apache Kafka, Hadoop (HDFS and YARN), MLflow, Redis, Grafana, Docker, Spark MLlib, Parquet.

### Architecture and responsibilities described
- **Ingestion and streaming:** Kafka ingestion and distributed micro-batch processing with PySpark Structured Streaming on YARN.
- **Time-based processing:** Sliding-window aggregations with watermarking.
- **Model tracking and inference:** MLflow tracking and low-latency scoring using Spark MLlib.
- **Alerting and storage:** Fraud alerts published to Redis and data archived in HDFS using Parquet.
- **Monitoring:** Grafana dashboard.

### Not documented
The CV does not provide throughput, end-to-end latency, fraud dataset details, model metrics, or proof of production deployment. Do not invent quantitative results.

---

## 3. Car Price Prediction — AI_MODEL-Cars-predictor

**Evidence:** Public GitHub repository.

- **Repository:** https://github.com/Seiflegnioui/AI_MODEL-Cars-predictor
- **Overview:** Car-price prediction project using data scraped from Avito.
- **Workflow described:** Scraping, data cleaning, exploratory data analysis, modeling, validation, and deployment.
- **Technologies described:** Python and Flask API.
- **Repository files described/visible:** Data, Model, View, `Main.py`, `Scraping_avito.py`, and `eda.py`.
- **Not documented:** A verified model score or a currently operating public deployment.

---

## 4. Human Resource Analytics — Data-Humain_Resource_Analytics

**Evidence:** Public GitHub repository.

- **Repository:** https://github.com/Seiflegnioui/Data-Humain_Resource_Analytics
- **Overview:** Employee-data analysis to study/predict attrition and support workforce-management decisions.
- **Workflow described:** Data cleaning, exploratory analysis, statistical tests, feature engineering, and machine-learning modeling.
- **Technologies described:** Python, pandas, scikit-learn, and visualization tools.
- **Repository files described/visible:** `HR_Analytics.csv` and `app.ipynb`.
- **Not documented:** A validated model score or production deployment.

---

## 5. Real-Time Chat Application — .NET Core 8 Server

**Evidence:** Public GitHub repository.

- **Repository:** https://github.com/Seiflegnioui/Chat-App-DOTNET-CORE8-ServerOnly
- **Overview:** Server-side real-time chat application using .NET Core 8.
- **Features described:** JWT authentication, WebSockets, and server-side controllers for chat functionality.
- **Language shown:** C#.

## 6. Real-Time Chat Application — React/Redux Client

**Evidence:** Public GitHub repository.

- **Repository:** https://github.com/Seiflegnioui/Chat-App-REACT-REDUX-ClientOnly
- **Overview:** Client-side real-time chat application using React and Redux with TypeScript.
- **Language shown:** TypeScript.

## 7. Gym Platform — React Client

**Evidence:** Public GitHub repository.

- **Repository:** https://github.com/Seiflegnioui/Gym-Platform-ReactJs-ClientSide
- **Overview:** Front-end for a fitness platform involving coaches, members, and delivery partners.
- **Features described:** Coaches can share programs and posts; members can like and comment.

## 8. Gym Platform — Node.js Server

**Evidence:** Public GitHub repository.

- **Repository:** https://github.com/Seiflegnioui/Gym-Platform-NodeJs-ServerSide
- **Overview:** Backend for the gym platform, including authentication, user management, and real-time communication.
- **Language shown:** JavaScript.

---

## 9. NeuroScan AI — Alzheimer’s Risk Screening Application (Contribution to confirm)

**Evidence:** Public LinkedIn post by a collaborator/teammate.

A public post describes NeuroScan AI as a web application intended to support early Alzheimer’s risk screening using an XGBoost model trained on clinical and genetic data from ADNI. The post mentions MMSE scores, age, sex, education level, APOE4 alleles, a Next.js/Tailwind frontend, and a Flask API. It reports 94% accuracy, but this metric is the post's claim and has not been independently verified here.

The post thanks Seif for insights into the ADNI dataset; it does not establish that Seif independently built the application or trained the model. The chatbot must describe this only as a collaborative project in which Seif contributed ADNI-dataset insights, unless he confirms additional responsibilities. Do not present medical predictions as diagnoses.

---

## 10. Credit-Card Fraud DataOps/MLOps Pipeline (Status and identity to confirm)

**Evidence:** Prior project discussions.

A separately discussed fraud pipeline used a Kaggle credit-card transaction dataset and included dlt ingestion, DuckDB raw storage, schema validation, dbt staging/marts/tests, Dagster orchestration, scikit-learn baselines, MLflow tracking, and model registration based on PR-AUC.

It is not yet clear whether this is a distinct project, an earlier iteration of the real-time banking fraud platform, or an experimental learning project. Do not count it as a separate completed portfolio project until Seif confirms its status and relationship to project 2.

---

## 11. Traffic-Light Optimization with Computer Vision (Learning / academic project; status to confirm)

**Evidence:** Prior project discussions.

A project concept/prototype involved YOLOv8 vehicle detection from a top-down road view, counting vehicles by road, and adjusting traffic-light green time based on traffic density and emergency-vehicle presence. A further idea involved capturing license plates when a red-light violation occurs.

Do not claim this is deployed or completed. Confirm current implementation status, dataset, detection results, and Seif's exact contribution before presenting it as a completed project.

---

## 12. Alzheimer’s Risk Prediction / Neuroimaging Research (Exploratory; status to confirm)

**Evidence:** Prior academic project discussions.

Seif previously explored an Alzheimer’s risk-prediction project using OASIS/ADNI-style data, FreeSurfer-derived brain-volume features, diagnosis labels, and APOE genotype, with feature selection and machine-learning modeling as intended components. A prior NITRC access request was discussed.

This should be described as an academic/research direction or exploratory project unless Seif confirms the dataset access, implementation, experiments, and results. Do not merge it automatically with NeuroScan AI; the available evidence does not establish they are the same project.
