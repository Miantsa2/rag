
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain_core.prompts import PromptTemplate

import streamlit as st


model_ollama = "mistral"

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


def get_llm(model_name):
    return Ollama(model=model_name, temperature=0.1)

def get_answer(query, model_name, k=4):

    # recuperation des chunks pertinents
    retrieved_docs = semantic_search(query, k=k)

    if not retrieved_docs:
        return "Aucun document indexe", []

    context_text = "\n\n---\n\n".join(
        f"[Source : {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    # Construction pprompt
    final_prompt = rag_prompt.format(context=context_text, question=query)

    llm = get_llm(model_name)
    answer = llm.invoke(final_prompt)

    return answer, retrieved_docs
