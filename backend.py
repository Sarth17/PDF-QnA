from pypdf import PdfReader
import re
import ollama
import math
import faiss
import pickle
import os
import numpy as np


#extracting text from pdf 
def extract_text(pdf_path):

    pages = []
    try:
        with open(pdf_path, 'rb') as pdf:

            reader = PdfReader(pdf)

            for i, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()

                    if page_text:
                        pages.append((i, page_text))
                    else:
                        print(f"page {i}: no text extracted")  

                except Exception as e:
                    print(f"page {i}: extraction error {e}")          

    except FileNotFoundError:
        return f"Error: file {pdf_path} not found"

    return pages


#all_pages_text = extract_text(pdf)



#cleaning it by removing whitespace and \n
def clean_text(pages):
    cleaned_pages = []

    for page_num , text in pages:

        clean_text = text.replace("\n", " ")
        clean_text = re.sub(r"\s+", " ", clean_text)

        cleaned_pages.append((page_num , clean_text))

    return cleaned_pages



#making 170 words chunk
def chunk_text(pages, chunk_size = 170):
    chunk_arr = []
    

    for page_num , text in pages:
        words = text.split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunk_arr.append({
                "text": chunk,
                "page": page_num
            })

    return chunk_arr    

#chunks = chunk_text(clean_text(extract_text("A.pdf")))
#print(len(chunks))
#print(chunks[0]) 


def chunk_to_embedding(chunks):
    
    embedding_store = []

    for chunk in chunks:
        
        response = ollama.embeddings(
                model='nomic-embed-text', 
                prompt= chunk["text"]
            ) 
        
        embedding_store.append({
            "text":chunk["text"],
            "embedding": response["embedding"],
            "page": chunk["page"]
        })

    return embedding_store    



def save_index(index, metadata, index_path="index.faiss", meta_path="metadata.pkl"):

    faiss.write_index(index, index_path)

    #writing metadata in file using pickle
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


def load_index(index_path="index.faiss", meta_path="metadata.pkl"):
    index = faiss.read_index(index_path)

    with open (meta_path, 'rb') as f:
        metadata = pickle.load(f)

    return index, metadata


def build_faiss_index(embedding_store):

    vectors = np.array(
        [item["embedding"] for item in embedding_store],
        dtype="float32"
    )

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    metadata = [
        {"text": item["text"], "page": item["page"]}
        for item in embedding_store
    ]

    return index, metadata


def faiss_search(question_embedding, index, metadata, top_k=3):
    query = np.array([question_embedding], dtype="float32")
    distances, indices = index.search(query, top_k)

    results = []
    for i in indices[0]:
        results.append(metadata[i])

    return results


def question_embedding(question):

    response = ollama.embeddings(
        model = 'nomic-embed-text',
        prompt = question
    )

    return response["embedding"]

"""
pre faiss version:

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

def similarity_search(question_embedding, embedding_store, top_k=3):
    scored_chunks = []

    for item in embedding_store:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored_chunks.append({
            "text": item["text"],
            "score": score,
            "page": item["page"]
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]
"""

def generate_answer(question, top_chunks):

    context = "\n\n".join(chunk["text"] for chunk in top_chunks)

    prompt = f"""
                You are a helpful assistant.
                Answer the question using ONLY the context below.
                If the answer is not present in the context, say "I don't know".

                Context:
                {context}

                Question:
                {question}

                Answer:
                """

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    return response["message"]["content"]


def main():

    pdf = input("enter pdf: ")
    text = extract_text(pdf)

    #if returns error then stop
    if isinstance(text, str) and text.startswith("Error:"):
        print(text)
        return

    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text)
    
    #if already exists show that
    if os.path.exists("index.faiss") and os.path.exists("metadata.pkl"):

        print("Loading existing embeddings...")
        index, metadata = load_index()

    #if does not exist then create 
    else:
    
        print("Building embeddings...")
        embedding_store = chunk_to_embedding(chunks)
        index, metadata = build_faiss_index(embedding_store)
        save_index(index, metadata)


    print("pdf indexed successfully")
    print("ask questions")
    print("type 'bye' to quit")

    #for multiple questions
    while True:

        question = input("ask question: ")

        if question.lower() == "bye":
            break

        question_vector = question_embedding(question)
        top_chunks = faiss_search(question_vector, index, metadata)
        answer = generate_answer(question, top_chunks)

        print("\nAnswer:")
        print(answer)
        pages = sorted(set(chunk["page"] for chunk in top_chunks))
        print("\nSources:")
        for p in pages:
            print(f"- Page {p}")




