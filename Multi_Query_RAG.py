import argparse
import os
# LangChain Community and Hub components
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from RAPTOR import *
from AnyFile_Loader import *
from langchain_community.chat_models import ChatOllama
from operator import itemgetter
from langchain.load import dumps, loads

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

#langsmith API key
os.environ['LANGCHAIN_API_KEY'] = ''
# this is optional, before using this line, create a project with this name in the langsmith
os.environ['LANGCHAIN_PROJECT']=''

os.environ['OPENAI_API_KEY'] = ''  # Ensure this is set securely
embd = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

def process_documents(source_directory: str, ignored_files: List[str] = []) -> List[str]:
    print("="*30)
    print("Initiating document processing...")
    documents = load_documents(source_directory, ignored_files)
    texts = [doc.page_content for doc in documents]
    print("Document processing completed. length: ", len(texts))
    print("="*30)
    return [text.replace("\n", " ") for text in texts]

def build_vectorstore_with_summaries(texts: List[str], n_levels: int = 3) -> Chroma:
    print("="*30)
    print("Initiating vectorstore building with summaries...")
    raptor_results = recursive_embed_cluster_summarize(texts, level=1, n_levels=n_levels)
    
    all_texts = texts.copy()
    for level in sorted(raptor_results.keys()):
        summaries = raptor_results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)

    vectorstore_path = os.environ.get('VECTORSTORE_PATH', 'Vec_Store')
    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd, persist_directory=vectorstore_path)
    print("Vectorstore building completed.")
    print("="*30)
    return vectorstore

def setup_ollama_language_model_chain(vectorstore: Chroma, LLM_name: str, topk: int):
    print(">>>chaining model:", LLM_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": topk})
    llm = ChatOllama(model=LLM_name, temperature=0)
    template = """
                Answer the question comprehensively and with detailed logical points based on the following context:
                {context}

                Question: {question}

                Start with a brief introduction to the topic, addressing the key elements of the question. Then, proceed with a detailed analysis, breaking down each component of the question into separate, well-thought-out points. Ensure that each point is supported by logical reasoning and relates back to the context provided. Conclude with a summary that synthesizes the findings and reflects on the implications or outcomes.

                Let's work this out step by step to ensure a thorough and well-structured answer.
                """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        print(f"Formatting docs: {docs}")
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print(">>>Chain setup completed.")
    print("="*30)
    return rag_chain

# Multi Query: Different Perspectives
template_mq = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template_mq)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(docs: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in docs for doc in sublist]
    unique_docs = list(set(flattened_docs))
    ud =  [loads(doc) for doc in unique_docs]
    print("Printing Uniques:",ud)
    return ud

def setup_language_model_chain(vectorstore: Chroma, topk: int):
    print(">>>Setting up LLM chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": topk})
    print(topk)

    def debug_retriever(documents):
        print(f"Documents received by retriever: {documents}")
        return documents

    retrieval_chain = generate_queries | retriever.map()| get_unique_union

    template = """
                Answer the question comprehensively and with detailed logical points based on the following context:
                {context}

                Question: {question}

                Start with a brief introduction to the topic, addressing the key elements of the question. Then, proceed with a detailed analysis, breaking down each component of the question into separate, well-thought-out points. Ensure that each point is supported by logical reasoning and relates back to the context provided. Conclude with a summary that synthesizes the findings and reflects on the implications or outcomes.

                Let's work this out step by step to ensure a thorough and well-structured answer.
                """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        print(f"Formatting docs: {docs}")
        formatted_docs = "\n\n".join(doc.page_content for doc in docs)
        print(f"Formatted docs: {formatted_docs}")
        return formatted_docs

    rag_chain = (
        {"context": retrieval_chain, "question": lambda x: x["question"]}
        | prompt
        | model
        | StrOutputParser()
    )
    print(">>>Chain setup completed.")
    print("="*30)
    return rag_chain

def invoke_chain(chain, question):
    print("Invoking the RAG chain...")
    try:
        response = chain.stream({"question": question})
        response = list(response)  # Convert generator to list to print all at once
        print(f"Chain response: {response}")
        for item in response:
            print(f"Response item: {item}, Type: {type(item)}")
        print("Chain invocation completed.")
        print("="*30)
        return response
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return "Error processing your request."


def load_vectorstore(path: str, embedding_function) -> Chroma:
    print(f">>>Loading vectorstore.")
    vectorstore = Chroma(persist_directory=path, embedding_function=embedding_function)
    print(">>>Vectorstore loaded.")
    print("="*30)
    return vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and query documents using a vectorstore.')
    parser.add_argument('--update', action='store_true', help='Flag to update the vectorstore (default action)')
    parser.add_argument('--no-update', dest='update', action='store_false', help='Flag to not update the vectorstore and load from existing path')
    parser.set_defaults(update=True)

    args = parser.parse_args()

    # Define the embedding function
    embd = OpenAIEmbeddings()
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'Melanoma_Papers')
    vectorstore_path = os.environ.get('VECTORSTORE_PATH', 'Vec_Store')

    if args.update:
        print("Update flag is set to True.")
        texts = process_documents(source_directory)
        vectorstore = build_vectorstore_with_summaries(texts, n_levels = 5)
        # Assume persist is correctly implemented
    else:
        print("Update flag is set to False.")
        vectorstore = load_vectorstore(vectorstore_path, embedding_function=embd)

    rag_chain = setup_language_model_chain(vectorstore)

    print("Application initialization completed. Type 'exit' to quit the application.")
    print("="*30)
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() == 'exit':
            break

        output = invoke_chain(rag_chain, user_input)
        print("Response:")
        print(output)