import os
import pypdf
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Any

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformer embedding function
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for input text
        """
        if isinstance(input, str):
            input = [input]
        
        return [embedding.tolist() for embedding in self.model.encode(input)]

def process_pdf(file: str) -> Dict[str, str]:
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

def process_folder(folder: str) -> None:
    """Processes all resumes in a folder and stores embeddings in ChromaDB."""
    # Initialize embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="resumes")

    # Find PDF files
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

        try:
            # Generate embedding explicitly
            resume_vector = model.encode(resume_data["text"], convert_to_numpy=True).tolist()
            
            # Print embedding details for verification
            print(f"✅ Embedding shape: {len(resume_vector)}")
            print(f"🧠 First 5 values: {resume_vector[:5]}")

            # Add to ChromaDB collection with explicit embedding
            collection.add(
                ids=[resume_data["file_name"]],
                embeddings=[resume_vector],
                documents=[resume_data["text"]],
                metadatas=[{
                    "file_name": resume_data["file_name"],
                    "file_path": resume_data["file_path"],
                }]
            )
            print(f"✅ Successfully processed {pdf_file}")
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")

    print("\n✅ All resumes have been processed and stored in ChromaDB!")

def query_resumes(query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Query similar resumes with full document retrieval
    """
    # Initialize embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="resumes")

    # Generate query embedding explicitly
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    # Perform query with full document retrieval
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Process and print results
    print("\n🔍 Query Results:")
    for i, (file_name, doc, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    ), 1):
        print(f"\n{i}. Resume: {file_name}")
        print(f"   Distance: {distance:.4f}")
        print(f"   Path: {metadata['file_path']}")
        # Print first 300 characters of document
        print(f"   Preview: {(doc[:300] + '...') if doc else 'No text'}")
    
    return results

# Run function
if __name__ == "__main__":
    resume_folder = "Resume/data/36893"  # Change to actual path
    process_folder(resume_folder)

    # Example query
    query_results = query_resumes("machine learning engineer")