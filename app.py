import os
import shutil
import tempfile

import streamlit as st

#   chargement de documents 
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

#  decoupage (chunking) 
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

#  embeddings locaux 
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

#  base vectorielle locale
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

#  LLM local  Ollama 
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain_core.prompts import PromptTemplate


# Dossier de persistance de la base vectorielle Chroma local
PERSIST_DIR = "./chroma_db"

# Modele d'embeddings local 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Nom du modele Ollama 
DEFAULT_OLLAMA_MODEL = "mistral"

# Parametres de chunking

# assez grand pour conserver quelques phrases / un paragraphe, assez petit pour rester dans la fenetre de contexte du LLM local avec plusieurs chunks assembles.
CHUNK_SIZE = 1000
# evite de couper une idee en plein milieu
CHUNK_OVERLAP = 150



# ETAPE 2 : PIPELINE D'INGESTION
# extraction
def load_file(uploaded_file, tmp_dir):

    # Sauvegarde temporairement le fichier uploade 
    file_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extension = uploaded_file.name.lower().split(".")[-1]

    # choix du DocumentLoader LangChain adapte a son extension.

    if extension == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif extension == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Extension non supportee : .{extension}")

    documents = loader.load()


    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    # liste de Document LangChain avec .page_content et .metadata

    return documents

# Chunking (Decoupage) 
def chunk_documents(documents):
    #  couper d'abord aux frontieres naturelles du texte (paragraphes, puis phrases, puis mots),et donne des chunks plus semantiquement coherents.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)

# Transformez les segments en vecteurs (Embeddings) 
def get_embeddings():
    if st.session_state.embeddings is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    return st.session_state.embeddings

# indexation et stockage
def index_files(uploaded_files):

    all_chunks = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            #  Extraction
            documents = load_file(uploaded_file, tmp_dir)
            # Chunking
            chunks = chunk_documents(documents)
            all_chunks.extend(chunks)

            if uploaded_file.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(uploaded_file.name)

    if not all_chunks:
        return 0

    # Vectorisation + stockage dans ChromaDB 
    embeddings = get_embeddings()

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
    else:
        st.session_state.vectorstore.add_documents(all_chunks)

   
    if hasattr(st.session_state.vectorstore, "persist"):
        st.session_state.vectorstore.persist()

    return len(all_chunks)



# ETAPE 3 : MODE sans LLM
# Interroge la base vectorielle et retourne les k chunks les plus proches semantiquement de la requete
def semantic_search(query, k=4):
    if st.session_state.vectorstore is None:
        return []
    results = st.session_state.vectorstore.similarity_search(query, k=k)
    return results


# ETAPE 4 : MODE avec LLM 

# Prompt :
PROMPT_TEMPLATE = """Tu es un assistant qui repond STRICTEMENT a partir du contexte fourni ci-dessous, extrait des documents de l'utilisateur.

Regles imperatives :
- Utilise UNIQUEMENT les informations presentes dans le contexte.
- Si la reponse ne se trouve pas dans le contexte, reponds explicitement :
  "Je ne trouve pas cette information dans les documents fournis."
- Ne fais aucune supposition ni ajout d'information externe.
- Reponds de maniere claire et concise, en français.

Contexte :
{context}

Question :
{question}

Reponse :"""

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

# instance de LLM Local via Ollama
def get_llm(model_name):
    return Ollama(model=model_name, temperature=0.1)


def get_answer(query, model_name, k=4):

    # recuperation des chunks pertinents
    retrieved_docs = semantic_search(query, k=k)

    if not retrieved_docs:
        return "Aucun document . Veuillez indexer des documents avant de poser une question", []

    context_text = "\n\n---\n\n".join(
        f"[Source : {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    # Construction pprompt
    final_prompt = rag_prompt.format(context=context_text, question=query)

    llm = get_llm(model_name)
    answer = llm.invoke(final_prompt)

    return answer, retrieved_docs



# ETAT DE SESSION 

def init_session_state():
    # Initialise les variables de session si non existantes
    if "chat_history" not in st.session_state:
        # Historique de la conversation 
        st.session_state.chat_history = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "indexed_files" not in st.session_state:
        # Suivi des fichiers deja indexe pour eviter les doublons,
        st.session_state.indexed_files = []

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

# ETAPE 1 : Initialisation  Squelette de l'Interface

def main():
    st.set_page_config(page_title="RAG Local - Clone NotebookLM", layout="wide")
    init_session_state()

    st.title(" Bienvenue sur Assistant RAG Local")
    st.caption("Votre assistant personnel pour interroger vos documents en local")


    # SIDEBAR 
  
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Documents")

        uploaded_files = st.file_uploader(
            "Charger vos fichiers",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        # bouton d'indexation 
        if st.button(" Indexer les documents", use_container_width=True):
            if uploaded_files:
                with st.spinner("Extraction, decoupage et vectorisation en cours..."):
                    n_chunks = index_files(uploaded_files)
                st.success(f"{n_chunks} chunks indexes avec succes")
            else:
                st.warning("Veuillez d'abord charger au moins un fichier")

        if st.session_state.indexed_files:
            st.caption("Fichiers indexes :")
            for fname in st.session_state.indexed_files:
                st.markdown(f"- {fname}")

        if st.button("Reinitialiser la base", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
            st.success("Base vectorielle reinitialisee")

    #    bouton d'activation/desactivation du LLM. 
        st.divider()
        st.subheader("Mode")
        llm_enabled = st.toggle("Activer LLM", value=True)
        if llm_enabled:
            st.info("Mode **RAG complet** : le LLM repond a partir des documents")
        else:
            st.info("Mode **Recherche Semantique pure** : aucun LLM, extraits bruts")

  
    # ZONE PRINCIPALE 
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(" Extraits  utilises du document"):
                    for i, doc in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Extrait {i} — source : `{doc.metadata.get('source', 'inconnue')}`**")
                        st.text(doc.page_content)
                        st.markdown("---")

    # Zone de saisie utilisateur
    user_query = st.chat_input("Posez une question sur vos documents...")

    if user_query:
        # Affichage du message utilisateur
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if st.session_state.vectorstore is None:
            with st.chat_message("assistant"):
                st.warning("Aucun document indexe. Merci de charger et indexer des documents dans la barre laterale.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Aucun document indexe.",
                "sources": [],
            })
        else:
            with st.chat_message("assistant"):
                if llm_enabled:
                    # avec LLM 
                    with st.spinner("Generation de la reponse par le LLM local..."):
                        answer, sources = get_answer(user_query, DEFAULT_OLLAMA_MODEL)
                    st.markdown(answer)

                    if sources:
                        with st.expander("Extraits sources utilises"):
                            for i, doc in enumerate(sources, start=1):
                                st.markdown(f"**Extrait {i} — source : `{doc.metadata.get('source', 'inconnue')}`**")
                                st.text(doc.page_content)
                                st.markdown("---")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    # sans LLM 
                    with st.spinner("Recherche semantique en cours..."):
                        results = semantic_search(user_query)

                    if not results:
                        st.markdown("Aucun resultat trouve")
                        content = "Aucun resultat trouve"
                    else:
                        content_lines = []
                        for i, doc in enumerate(results, start=1):
                            source_name = doc.metadata.get("source", "inconnue")
                            st.markdown(f"**Resultat {i} — source : `{source_name}`**")
                            st.text(doc.page_content)
                            st.markdown("---")
                            content_lines.append(f"[{source_name}] {doc.page_content}")
                        content = "\n\n".join(content_lines)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": content,
                        "sources": [],  #  affiches bruts ci-dessus
                    })



if __name__ == "__main__":
    main()