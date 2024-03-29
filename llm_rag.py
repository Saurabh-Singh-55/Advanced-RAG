from Imports import *

# Importing the necessary functions from RAPTOR.py and AnyFile_Loader.py
from RAPTOR import recursive_embed_cluster_summarize
from AnyFile_Loader import load_documents

# Initialize the embeddings and model
embd = OpenAIEmbeddings()
embedding = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Set the LangChain environment variables
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="ls__6a928a2b77fd42819b9b194ac3a9f5a4"

def process_documents(source_directory: str, ignored_files: List[str] = []) -> List[str]:
    """
    Load and process documents from the specified directory.
    """
    documents = load_documents(source_directory, ignored_files)
    texts = [doc.page_content for doc in documents]
    return [text.replace("\n", " ") for text in texts]

def build_vectorstore_with_summaries(texts: List[str], n_levels: int = 3) -> Chroma:
    """
    Process texts through RAPTOR clustering and build a Chroma vectorstore with the resulting summaries.
    """
    raptor_results = recursive_embed_cluster_summarize(texts, level=1, n_levels=n_levels)
    
    all_texts = texts.copy()
    for level in sorted(raptor_results.keys()):
        summaries = raptor_results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)

    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
    return vectorstore

def setup_language_model_chain(vectorstore: Chroma):
    """
    Setup a language model chain for generating responses using the provided vectorstore.
    """
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    # Define the source directory for document loading
    source_directory = 'path/to/document/source'

    # Load and process documents
    texts = process_documents(source_directory)

    # Build vectorstore with summaries from RAPTOR clustering
    vectorstore = build_vectorstore_with_summaries(texts)

    # Setup and use the LLM chain with the built vectorstore
    rag_chain = setup_language_model_chain(vectorstore)

    # Example of invoking the chain with questions
    print(rag_chain.invoke("Give Guidelines to diagnose Melanoma"))
    print(rag_chain.invoke("Give Guidelines to treat Melanoma"))
