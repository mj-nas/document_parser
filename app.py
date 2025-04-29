import os
import pypdf
import chromadb
from fastapi import FastAPI, Request, Query, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from tqdm import tqdm

app = FastAPI()

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collections = chroma_client.list_collections()
print("Collections:", collections)
# tenant = chroma_client.tenant
# print("Tenant:", tenant)
collection = chroma_client.get_or_create_collection(name="resumes")

# Jinja2 Templates (for UI rendering)
templates = Jinja2Templates(directory="templates")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the resume search UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_resumes(request: Request, query: str = Form(...)):
    # Convert query to vector embedding
    query_vector = generate_embedding(query)
    # Query for similar embeddings
    print(f"🔍 Searching for resumes similar to: {query}")
    print("Query Vector:", query_vector[:5])
    results = query_embedding(query_vector)
    # Print results for debugging
    print("Query Results:", results)
    # Render results in the UI

    return templates.TemplateResponse("index.html", {
        "request": request,
        "query": query,
        "results": results
    })

RESUME_FOLDER = "Resume/samples"

# Serve the "resumes" folder as a static file directory
app.mount("/files", StaticFiles(directory=RESUME_FOLDER), name="files")

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    """Serve a resume PDF file."""
    file_path = os.path.join(RESUME_FOLDER, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    return {"error": "File not found"}

@app.post("/generate_embedding")
async def generate_embedding_endpoint():
    resume_folder = "Resume/samples"  # Change to actual path
    process_folder(resume_folder)
    

def process_pdf(file):
    """Extracts text from a PDF resume."""
    file_path = os.path.abspath(file)
    file_name = os.path.basename(file)
    
    try:
        with open(file, "rb") as f:
            reader = pypdf.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ Error reading PDF file {file_name}: {e}")
        text = ""

    return {
        "file_path": file_path,
        "file_name": file_name,
        "text": text
    }

def generate_embedding(text):
    """Generates a vector embedding from text using MiniLM."""
    if not text.strip():  # Ensure text is not empty
        print("⚠️ Skipping empty resume text for embedding.")
        return None  
    return model.encode(text, convert_to_numpy=True).tolist()  # Convert to list

def process_folder(folder):
    """Processes all resumes in a folder and stores embeddings in ChromaDB."""
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDF files found in folder: {folder}")
        return
    
    print(f"🔄 Processing {len(pdf_files)} resumes...\n")

    for pdf_file in tqdm(pdf_files, desc="Processing Resumes", unit="files"):
        pdf_path = os.path.join(folder, pdf_file)

        print(f"📄 Processing {pdf_file}")
        resume_data = process_pdf(pdf_path)

        if not resume_data["text"]:
            print(f"⚠️ Skipping {pdf_file} (No text extracted)\n")
            continue

        print(f"📄 {pdf_file} text:\n{resume_data['text'][:500]}...\n")
        resume_vector = model.encode(resume_data["text"], convert_to_numpy=True).tolist()
        print(f"✅ Embedding shape: {len(resume_vector)}")
        # resume_vector = generate_embedding(resume_data["text"])
        # print("✅ Embedding shape:", resume_vector.shape)
        print("🧠 First 5 values:", resume_vector[:5])

        if resume_vector is None:
            continue  # Skip if embedding failed

        # Store in ChromaDB
        collection.add(
            ids=[resume_data["file_name"]],
            embeddings=[resume_vector],
            metadatas=[{
                "file_name": resume_data["file_name"],
                "file_path": resume_data["file_path"],
            }]
        )

    print("\n✅ All resumes have been processed and stored in ChromaDB!")


def test_funtion():
    print("Testing function")
    text = "This is a test sentence."
    embedding = generate_embedding(text)

    collection.add(
            ids=["000001"],
            embeddings=[embedding],
            metadatas=[{
                "file_name": "resume_data[file_name]",
                "file_path": "resume_data[file_path]",
            }]
        )


def query_embedding(embedding):
    """Query ChromaDB for similar embeddings."""
    # Query for similar embeddings
    results = collection.query(query_embeddings=[embedding], n_results=5)
    # print("Query Results:", results)

    # # Get metadata for similar embeddings
    # for result in results:
    #     metadata = collection.get_metadata(result["id"])
    #     print("Metadata:", metadata)


     # Format results
    response = []
    for i, match in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        response.append({
            "rank": i + 1,
            "file_name": metadata["file_name"],
            "file_url": f"/files/{metadata["file_name"]}",
            "file_path": metadata["file_path"],
            "similarity_score": results["distances"][0][i]
        })

    return response

def query_resumes(path):
    """Query similar resumes with full document retrieval."""
    # extract text from jd_pdf file
    query_data = process_pdf(path)
    query_text = query_data["text"]
    # Generate query embedding explicitly
    query_vector = model.encode(query_text, convert_to_numpy=True).tolist()
    # Perform query with full document retrieval
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "distances"]
    )

    final_output = []
    for i, (file_name, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    ), 1):
        final_output.append({
        "file_name": file_name,
        "metadata": metadata,
        "distance": distance
        })

    return final_output
   
# result = query_resumes("Resume/jd/Req 1.pdf")
# print("Query Results:", result)

# Run function
# resume_folder = "Resume/samples"  # Change to actual path
# process_folder(resume_folder)

# test_funtion()
# print(collection.get(include=['embeddings', 'documents', 'metadatas']))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
