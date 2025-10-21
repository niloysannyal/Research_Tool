# 🔍 Research Tool

An intelligent **AI-powered research assistant** built with **LangChain**, **Streamlit**, **FAISS**, and **Hugging Face embeddings**.  
It allows you to **input URLs**, automatically **scrape and chunk their content**, **embed** the text for retrieval, and then **ask natural language questions** — getting answers backed by source references.

---

## 🚀 Features

- 🌐 **URL Data Extraction** — Load and process multiple web pages directly.  
- 🧠 **Semantic Search & Q&A** — Ask complex research questions and get accurate, referenced answers.  
- 💬 **Conversational Model Integration** — Powered by **Google Gemini 2.5 Flash** (via LangChain’s unified interface).  
- 💾 **Persistent FAISS Vector Store** — Automatically saves embeddings for reuse.  
- ⚡ **Streamlit Interface** — Simple and interactive web app for researchers, students, and analysts.  

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / Logic** | Python, LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | Hugging Face (`all-MiniLM-L6-v2` by default) |
| **LLM Provider** | Google Gemini (`gemini-2.5-flash`) |
| **Environment** | Conda / virtualenv |
| **Data Loader** | `UnstructuredURLLoader` from LangChain |

---

## 🏗️ Project Structure
```
Research_Tool/
├── app.py               # Main Streamlit application
├── stored_vectors/      # Folder for saved FAISS vector indices
│ └── vector_index.pkl
├── .env                 # Contains API keys (Google GenAI, etc.)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/niloysannyal/Research_Tool.git
cd Research_Tool
```
#### 2️⃣ Create a Virtual Environment
**Using Conda:**
```
conda create -n research_tool python=3.10 -y
conda activate research_tool
```
**Or with venv:**
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```
#### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
#### 4️⃣ Add Your API Key
```
GOOGLE_API_KEY=your_google_genai_api_key_here
```

---

## ▶️ Usage
#### 1️⃣ Run the App
```
streamlit run app.py
```
#### 2️⃣ In the Sidebar:
- Enter up to 3 source URLs
- Click “Process URL” to load and embed content
#### 3️⃣ In the Main Section:
- Type your question about the processed sources
- Get an AI-generated answer with source references

---

## 🧠 How It Works

1. **Load Data:** Uses `UnstructuredURLLoader` to scrape article content.  
2. **Split Text:** Breaks large text into manageable chunks with `RecursiveCharacterTextSplitter`.  
3. **Embed:** Generates numerical vector representations using `HuggingFaceEmbeddings`.  
4. **Store:** Saves embeddings in a **FAISS** index for efficient retrieval.  
5. **Ask Questions:** `RetrievalQAWithSourcesChain` fetches relevant chunks and queries the **Gemini LLM**.  
6. **Display:** Streamlit displays the **answer** and **sources** interactively.

---

## 📦 Example Output

**Question:**
> What are the key findings of the article?

**Answer:**
> The study concludes that hybrid models outperform traditional approaches in real-time language tasks.

**Sources:**
- https://example.com/article1  
- https://example.com/article2  

---

## 🔒 Environment Variables

| Variable | Description |
|-----------|-------------|
| `GOOGLE_API_KEY` | API key for Google Generative AI (Gemini) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork this repo and submit a pull request.

---

## 🧑‍💻 Author

**Niloy Sannyal**  
📍 Dhaka, Bangladesh  
📧 [niloysannyal@gmail.com](mailto:niloysannyal@gmail.com)  
🔗 [GitHub: niloysannyal](https://github.com/niloysannyal)

---

## 🪪 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

> “Empowering researchers with intelligent tools for smarter insights.”











