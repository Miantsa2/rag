# RAG Local — Clone de NotebookLM

Application Streamlit 100% locale de type RAG (Retrieval-Augmented Generation),
sans aucun appel à une API externe (OpenAI, etc.).

## 1. Installation

```bash
python -m venv venv
Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Pré-requis : Ollama

1. Installer [Ollama](https://ollama.com).
2. Télécharger un modèle et le laisser tourner en tâche de fond :
   ```bash
   ollama run mistral
   ```
   (ou `qwen2.5-coder`, auquel cas modifier `OLLAMA_MODEL_NAME` dans `app.py`).

## 3. Lancement de l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur (par défaut http://localhost:8501).

## 4. Utilisation

1. Dans la barre latérale, charger un ou plusieurs fichiers (`.pdf`, `.txt`, `.md`).
2. Cliquer sur **Indexer les documents** : extraction → découpage → vectorisation
   → stockage dans ChromaDB (persisté localement dans un dossier temporaire).
3. Basculer le toggle **Assistant RAG complet** :
   - **Désactivé** → mode *Recherche sémantique pure* : la question est
     directement comparée aux embeddings des chunks, sans aucun LLM. Les
     extraits bruts et leur fichier source sont affichés.
   - **Activé** → mode *RAG complet* : les extraits pertinents sont injectés
     comme contexte dans un prompt strict envoyé au LLM local (Ollama), qui
     doit répondre uniquement à partir de ce contexte. Les extraits utilisés
     restent consultables sous la réponse (transparence).

## 5. Choix techniques

- **Chunking** : `chunk_size=1000`, `overlap=150` caractères (voir
  justification commentée dans `app.py`, section *Stratégie de chunking*) :
  compromis entre cohérence sémantique d'un chunk et pertinence de la
  recherche par similarité, avec un chevauchement qui limite la perte
  d'information aux frontières de segments.
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` via
  `HuggingFaceEmbeddings` — léger, rapide sur CPU, multilingue suffisant pour
  un usage local de TP.
- **Base vectorielle** : ChromaDB, persistée sur disque.
- **Génération** : LLM local via Ollama (`Ollama` de `langchain_community`),
  aucune donnée envoyée à l'extérieur.

## 6. Structure du projet

```
.
├── app.py             # Application complète (commentée)
├── requirements.txt    # Dépendances Python
└── README.md
```
