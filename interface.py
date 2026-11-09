
import streamlit as st

import ingestion
import rag
import os
import shutil

# ETAT DE SESSION 
def init_session_state():
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
                    n_chunks = ingestion.index_files(uploaded_files)
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
            if os.path.exists(ingestion.base_chroma):
                shutil.rmtree(ingestion.base_chroma)
            st.success("Base vectorielle reinitialisee")

    #    bouton d'activation/desactivation du LLM. 
        st.divider()
        st.subheader("Mode")
        llm_enabled = st.toggle("Activer LLM", value=True)
       
    # ZONE PRINCIPALE 
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"],avatar="user.jpg" if msg["role"] == "user" else "bot.jpg"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(" Extraits utilises du document"):
                    for i, doc in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Extrait {i} — source : `{doc.metadata.get('source', 'inconnue')}`**")
                        st.text(doc.page_content)
                        st.markdown("---")

    # Zone de saisie utilisateur
    user_query = st.chat_input("Posez une question...")

    if user_query:
        # Affichage du message utilisateur
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user",avatar="user.jpg"):
            st.markdown(user_query)

        if st.session_state.vectorstore is None:
            with st.chat_message("assistant",avatar="bot.jpg"):
                st.warning("Aucun document indexe. ")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Aucun document indexe.",
                "sources": [],
            })
        else:
            with st.chat_message("assistant",avatar="bot.jpg"):
                if llm_enabled:
                    # avec LLM 
                    with st.spinner("Generation de la reponse par le LLM local..."):
                        answer, sources = rag.get_answer(user_query, rag.model_ollama)
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
                        results = rag.semantic_search(user_query)

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