import os
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


#CONFIGURATION
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

FILE_PATH = Path(
    r"C:\Users\VVDN\Desktop\Data Scientics\RAG"
    r"\extracted_text\Aspirin_solubility_enhancer.txt"
)

DB_PATH = Path(__file__).resolve().parent / "vector_db"
COLLECTION_NAME = "research_collection"

if not FILE_PATH.is_file():
    raise FileNotFoundError(
        f"Text file not found: {FILE_PATH}"
    )


#LOAD DOCUMENT

loader = TextLoader(
    str(FILE_PATH),
    encoding="utf-8"
)

documents = loader.load()

for document in documents:
    document.metadata["source_file"] = FILE_PATH.name

print("Text loaded successfully")
print("Number of documents:", len(documents))


#CHUNK DOCUMENT

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

for i, chunk in enumerate(chunks, start=1):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source_file"] = FILE_PATH.name

print("Chunking completed")
print("Total chunks:", len(chunks))


#LOAD EMBEDDING MODEL

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded")


#CREATE / REBUILD CHROMA DATABASE

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

existing_collections = [
    collection.name
    for collection in client.list_collections()
]

# This script rebuilds its own collection on each run.
# Other collections in the database are left untouched.
if COLLECTION_NAME in existing_collections:
    client.delete_collection(name=COLLECTION_NAME)

vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding
)

ids = [
    f"aspirin_enhancer_chunk_{i}"
    for i in range(len(chunks))
]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)

print("Chunks embedded and stored in ChromaDB")
print("Stored chunks:", len(chunks))


#6.CREATE RETRIEVER

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("Retriever created")


#CONFIGURE GROQ LLM

llm = ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    temperature=0,
    max_tokens=700,
    api_key=GROQ_API_KEY
)

print("LLM ready")


#PROMPT

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a scientific research-paper analysis assistant.

Use ONLY the retrieved evidence to answer questions.

STRICT RULES:

1. Never invent experimental values, units, or citations.
2. Every numerical result must be supported by the supplied
   retrieved text.
3. Identify the source filename and chunk ID supporting each
   numerical result.
4. Preserve the reported units, uncertainty, formulation ratio,
   temperature, medium, and measurement type.
5. Do not confuse dissolution percentage with solubility in mg/mL.
6. Do not combine measurements from different formulations or studies.
7. Do not claim that a prediction is experimentally validated
   merely because a paper discusses a similar formulation.
8. If the requested value is absent, state:
   "The retrieved evidence does not contain this value."
9. If sources conflict, report the conflict and do not silently
   choose one result.
10. Distinguish reported experimental findings from interpretation.
11. Do not infer experimental conditions absent from the text.

Before answering, check whether the retrieved passages contain
the exact evidence needed for the question.

Retrieved evidence:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


#RETRIEVAL + GENERATION

def ask_rag(question, debug=True):

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": (
                "No relevant documents were retrieved. "
                "Try rephrasing the question."
            ),
            "sources": []
        }

    # Display the exact evidence given to the LLM.
    if debug:
        print("\n" + "=" * 70)
        print("RETRIEVED EVIDENCE DEBUG")
        print("=" * 70)

        for rank, doc in enumerate(retrieved_docs, start=1):
            print(f"\nRank: {rank}")
            print("Source:", doc.metadata.get(
                "source_file", "unknown"
            ))
            print("Chunk ID:", doc.metadata.get(
                "chunk_id", "unknown"
            ))
            print("Start index:", doc.metadata.get(
                "start_index", "unknown"
            ))
            print("Retrieved text:")
            print(doc.page_content)
            print("-" * 70)

    # Prepare context for the LLM.
    context_parts = []

    for doc in retrieved_docs:
        source = doc.metadata.get(
            "source_file", "unknown"
        )
        chunk_id = doc.metadata.get(
            "chunk_id", "unknown"
        )
        start_index = doc.metadata.get(
            "start_index", "unknown"
        )

        context_parts.append(
            f"SOURCE: {source}\n"
            f"CHUNK ID: {chunk_id}\n"
            f"START INDEX: {start_index}\n"
            f"TEXT:\n{doc.page_content}"
        )

    context = "\n\n--- EVIDENCE SEPARATOR ---\n\n".join(
        context_parts
    )

    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": retrieved_docs
    }


#INTERACTIVE QUESTION LOOP

if __name__ == "__main__":

    print("\nRAG pipeline is ready.")
    print("Type 'exit' to quit.")

    while True:
        question = input(
            "\nEnter your research question: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Thanks for using the RAG system.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = ask_rag(question, debug=True)

            print("\n" + "=" * 70)
            print("LLM ANSWER")
            print("=" * 70)
            print(result["answer"])

            print("\nRETRIEVED SOURCES")
            for doc in result["sources"]:
                print(
                    f"Source: {doc.metadata.get('source_file', 'unknown')} | "
                    f"Chunk: {doc.metadata.get('chunk_id', 'unknown')} | "
                    f"Start index: {doc.metadata.get('start_index', 'unknown')}"
                )

        except Exception as error:
            print(f"An error occurred: {error}")
