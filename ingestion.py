
import os
import shutil
import streamlit as st
import tempfile


from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma




# Dossier de persistance de la base vectorielle Chroma local
base_chroma = "./chroma_db"
# Modele d'embeddings local 
model_embedding = "sentence-transformers/all-MiniLM-L6-v2"

# Parametres de chunking
chunk_size = 1000
chunk_overlap = 150



# extraction
def load_file(uploaded_file, tmp_dir):

    # Sauvegarde temporairement le fichier uploade 
    file_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extension = uploaded_file.name.lower().split(".")[-1]

    # choix du DocumentLoader
    if extension == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif extension == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Extension non supportee : .{extension}")

    documents = loader.load()


    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    return documents

# Chunking (Decoupage) 
def chunking(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ","\n", " ", "","\n\n",],
    )
    return splitter.split_documents(documents)


# Transformez les segments en vecteurs (Embeddings) 
def get_embeddings():
    if st.session_state.embeddings is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name=model_embedding
        )
    return st.session_state.embeddings

# indexation et stockage
def index_files(uploaded_files):
    tabs_chunks = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            #  Extraction
            documents = load_file(uploaded_file, tmp_dir)
            # Chunking
            chunks = chunking(documents)
            tabs_chunks.extend(chunks)

            if uploaded_file.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(uploaded_file.name)

    if not tabs_chunks:
        return 0

    embeddings = get_embeddings()
    
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=tabs_chunks,
            embedding=embeddings,
            persist_directory=base_chroma,
        )
    else:
        st.session_state.vectorstore.add_documents(tabs_chunks)
    if hasattr(st.session_state.vectorstore, "persist"):
        st.session_state.vectorstore.persist()

    return len(tabs_chunks)
    



