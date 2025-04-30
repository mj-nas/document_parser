# 🧠 AI-Powered Resume Shortlisting with Retrieval-Augmented Generation (RAG)

This is a prototype application that demonstrates how AI and modern NLP techniques can enhance candidate recommendation in staffing platforms like **JobzMachine**. It uses **Retrieval-Augmented Generation (RAG)** with **domain-adapted embeddings**, vector similarity search, and semantic query processing to rank and retrieve resumes based on job descriptions or free-text queries.

---

## 🚀 Technologies Used

- 🐍 **Python 3**
- ⚡ **FastAPI** – For serving the UI and backend APIs
- 🧠 **SentenceTransformers (all-MiniLM-L6-v2)** – For generating dense vector embeddings
- 🧾 **ChromaDB** – Open-source vector database for fast similarity search
- 🧠 **Semantic Search / RAG** – Retrieve top-matching resumes using dense vector embeddings
- 📄 **PDF Parsing** – Extract resume content using `pypdf`
- 🌐 **Jinja2 Templates** – For UI rendering
- 📦 **tqdm** – For progress bars in terminal
- 📊 **LLM Foundations** – Implements core principles of Retrieval-Augmented Generation (RAG)

---

## 🛠️ How to Run the Application

1. **Install Dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Start ChromaDB (Separate Terminal)**

   ```bash
   chroma run --path ./chroma_storage
   ```

3. **Process Resume Folder (Optional)**

   In `app.py`, call `process_folder("Resume/samples")` once to add your data to the vector DB.

4. **Run the FastAPI Server**

   ```bash
   python app.py
   ```

5. **Access the App**

   Open your browser and go to: `http://127.0.0.1:8001`

---

## 🧭 How It Works (Architecture Diagram)

This application follows a simple semantic retrieval pipeline using ChromaDB + MiniLM:

![Semantic Resume Search Architecture](./A_flowchart-style_2D_digital_diagram_depicts_a_Res.png)

---

## 📌 Notes

- This prototype enables **semantic filtering** of resumes rather than just keyword search.
- Easily extendable to include **LLM-based summarization** or **chat-based resume agents**.
- Can evolve into a full Retrieval-Enhanced LLM pipeline.

---

## 💡 What's Next?

- Integrate with OpenAI GPT or LLaMA for full RAG-based response generation.
- Deploy as microservices in a production-ready pipeline.
- Add candidate feedback loops for fine-tuning rankings.

---

> Built as a hands-on learning project into the world of AI/ML, LLMs, vector databases, and domain-adapted retrieval systems 🎓✨
