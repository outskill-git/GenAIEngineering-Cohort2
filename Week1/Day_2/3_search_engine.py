from fastapi import FastAPI
import os
import pandas as pd
from PyPDF2 import PdfReader
import re
import glob
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()
faiss_index = None

os.makedirs(os.path.join('csv_files'), exist_ok=True)

def chunk_pdf_to_dataframe(pdf_path, num_chunks = 5):
    reader = PdfReader(pdf_path)
    data=[]
    filename = os.path.basename(pdf_path)
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text.strip():
            continue
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        actual_num_chunks = min(num_chunks, len(text))
        if actual_num_chunks <= 1:
            data.append({'filename': filename,'page_number': page_num,'chunk_number': 1,'chunk': text})
        else:
            chunk_size = len(text) // actual_num_chunks
            for i in range(actual_num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(text))
                if i == actual_num_chunks - 1:
                    end = len(text)
                chunk = text[start:end]
                data.append({'filename': filename,'page_number': page_num,'chunk_number': i + 1,'chunk': chunk})
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join('csv_files', filename.replace('.pdf','.csv')), index=False)

@app.post("/chunk_pdf")
async def chunk_pdf(pdf_file_path,num_chunks= 5):
    chunk_pdf_to_dataframe(pdf_file_path, num_chunks=int(num_chunks))

    global faiss_index
    faiss_index=build_faiss_index()
    return {"status": "success","message": f"PDF chunked successfully","file_path": pdf_file_path}



def embed_text_chunks(chunks, embedding_model_name="all-MiniLM-L6-v2"):

    model = SentenceTransformer(embedding_model_name)
    # print(model)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    # print(embeddings)
    return embeddings

# Function to build a FAISS index
def build_faiss_index():
    files = glob.glob(os.path.join('csv_files', "*.csv"))
    print(files)
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(len(df))
    embedding_model_name = "all-MiniLM-L6-v2"
    embeddings = embed_text_chunks(list(df['chunk']), embedding_model_name)
    print('embeddings completed')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print('files indexed')
    return index



# @app.post("/search_chunks")
# def search_chunks(search_string):
#     search_list = search_string.split()
#     files = glob.glob(os.path.join('Day_1','csv_files', "*.csv"))
#     df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
#     matches_df = df[df['chunk'].str.contains('|'.join(search_list), case=False, na=False)] #key word search
#     return matches_df.to_json()

@app.post("/search_chunks")
def search_chunks(search_string):
    query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([search_string], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k=5)
    files = glob.glob(os.path.join('csv_files', "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    matches_df=df.iloc[indices[0]]
    return matches_df.to_json()

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

def main():
    global faiss_index
    faiss_index=build_faiss_index()

if __name__ == "__main__":
    print('starting services')
    main()
    uvicorn.run(app, host="0.0.0.0", port=9321)
