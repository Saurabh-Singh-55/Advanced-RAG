from AnyFile_Loader import *
from Multi_Query_RAG import *
import streamlit as st
import os
import io
import sys
import openai
import ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
api_key = ''

#################################################################################################################    
##################################   Util Functions   ###########################################################
#################################################################################################################
load_dotenv()
api_key = ''
class CapturePrints:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.captured_output = io.StringIO()

    def __enter__(self):
        sys.stdout = self.captured_output
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = sys.__stdout__
        if self.log_callback:
            self.log_callback(self.captured_output.getvalue())

# Initialize the log in the session state if it doesn't exist
if 'log' not in st.session_state:
    st.session_state.log = ""

def update_log(message):
    st.session_state.log += message

# App title
st.set_page_config(page_title="Document Query Interface")

def extract_pages(source_directory: str):
    """ Extracts pages from documents located in the specified directory using the load_documents function.
        Args: source_directory (str): The directory containing the files to be processed.
        Returns:list: A list of strings, each representing the content of a page from the extracted documents.
    """
    print("="*30)
    print(f">>>Extracting from: {source_directory}")

    # Load documents using the function from AnyFile_loader.py
    documents = load_documents(source_directory)
    if not documents:
        print("No documents found or loaded.")
        return []

    # Add file path and title to metadata for each document
    for doc in documents:
        if 'file_path' not in doc.metadata:
            doc.metadata['file_path'] = source_directory
        if 'title' not in doc.metadata:
            doc.metadata['title'] = os.path.basename(doc.metadata['file_path']).replace('_', ' ').replace('.pdf', '')

    # Extract page content from each document
    extracted_pages = [doc.page_content for doc in documents]
    print(f">>>Extracted {len(extracted_pages)} pages.")
    print("="*30)
    return documents

def store_embeddings(documents: List[str], chunk_size: int, chunk_overlap: int, level: int, n_levels: int):
    """ Creates and stores embeddings from the provided texts.
        Args:
        texts (List[str]): The texts to process and create embeddings for.
        chunk_size (int): Size of the text chunk for processing.
        chunk_overlap (int): Overlap size between chunks.
        level (int): Starting level for processing.
        n_levels (int): Number of levels for hierarchical processing.
        Returns:
        Chroma: The created vector store.
    """
    print(">>>Creating and storing embeddings...")

    # Call the function to build the vector store with summaries
    vectorstore_path = os.environ.get('VECTORSTORE_PATH', 'Vec_Store')
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=False,
    )
    docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(docs, embedding=embd, persist_directory=vectorstore_path)
    print(">>>Raw Embeddings stored.")
    print("="*30)
    return vectorstore

def create_store_embeddings(texts: List[str], documents: List[str], chunk_size: int, chunk_overlap: int, level: int, n_levels: int):
    """ Creates and stores embeddings from the provided texts.
        Args:
        texts (List[str]): The texts to process and create embeddings for.
        chunk_size (int): Size of the text chunk for processing.
        chunk_overlap (int): Overlap size between chunks.
        level (int): Starting level for processing.
        n_levels (int): Number of levels for hierarchical processing.
        Returns:
        Chroma: The created vector store.
    """
    print(">>>Creating and storing embeddings...")

    # Call the function to build the vector store with summaries
    vectorstore_path = os.environ.get('VECTORSTORE_PATH', 'Vec_Store')
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1500,
        chunk_overlap=50,
        keep_separator=False,
    )
    docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(docs, embedding=embd, persist_directory=vectorstore_path)
    print(">>>Summary Embeddings stored.")
    print("="*30)
    return vectorstore

def create_summary_tree(documents: List[str], chunk_size: int, chunk_overlap: int, level: int, n_levels: int) -> Dict:
    print(">>> Creating summary tree...")

    extracted_pages = [doc.page_content for doc in documents]
    texts = [text.replace("\n", " ") for text in extracted_pages]
    # Assuming recursive_embed_cluster_summarize is correctly implemented and imported
    raptor_results = recursive_embed_cluster_summarize(
        texts,
        level=level,
        n_levels=n_levels
    )

    all_texts = []
    for level in sorted(raptor_results.keys()):
        summaries = raptor_results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)

    print(">>> Summary tree created.")
    print("="*30)
    return all_texts

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validates the OpenAI API key by attempting to list the available models.

    Args:
    api_key (str): The OpenAI API key to validate.

    Returns:
    bool: True if the API key is valid, False otherwise.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.APIConnectionError:
        return False
    else:
        return True

#################################################################################################################    
##################################   Side Bar Interface    ######################################################
#################################################################################################################

with st.sidebar:
    st.title('Document Query Interface')

    llm_type = st.sidebar.radio("Choose LLM Type", ['API', 'Local'])

    if llm_type == 'API':
        api_key = st.sidebar.text_input('Enter API key:', type='password', key='api_key')
        # Check if the API key is valid
        api_key_valid = validate_openai_api_key(api_key)
        
        if api_key_valid:
            st.sidebar.success('API key validated!', icon='✅')
            os.environ['OPENAI_API _KEY'] = api_key
            embd = OpenAIEmbeddings()
            model = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature=0, model="gpt-3.5-turbo")
        else:
            st.sidebar.error('Invalid API key!', icon='⚠️')
    elif llm_type == 'Local':
        list_response = ollama.list()
        # Extract the 'name' field from each model's details
        pulled_models = [model['name'] for model in list_response['models']]
        selected_model = st.sidebar.selectbox('Select a Model', list(reversed(pulled_models)))
    # Document Update and Processing Section
    st.markdown('---')
    st.subheader('Document Processing')
    update_enabled = st.checkbox("Update Documents", key='update_documents')

    if update_enabled:
        source_directory = st.text_input("Directory path containing PDFs:")
        if st.button("Load Documents"):
            if not os.path.exists(source_directory):
                st.error("Invalid directory path!")
            else:
                with CapturePrints(log_callback=update_log):
                    st.session_state.total_pages = extract_pages(source_directory)
                st.success("Documents loaded successfully.")
        
        col1, col2= st.columns([1,1])
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=1000, value=500, help='Size of text chunk for processing.')
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=100, value=50, help='Overlap size between chunks.')
        col1, col2= st.columns([1,1])
        with col1:
            level = st.number_input("Level", min_value=1, max_value=5, value=1, help='Starting level for processing.')
        with col2:
            n_levels = st.number_input("Number of Levels", min_value=1, max_value=10, value=3, help='Number of levels for hierarchical processing.')

        if st.button("Store Embeddings"):
            with CapturePrints(log_callback=update_log):
                st.session_state.vectorstore = store_embeddings(st.session_state.total_pages, chunk_size, chunk_overlap, level, n_levels)
            st.success("Embeddings stored.")
            st.success('VectorStore is Loaded')

        if st.button("Create Summary Tree"):
            with CapturePrints(log_callback=update_log):
                st.session_state.clusters = create_summary_tree(st.session_state.total_pages, chunk_size, chunk_overlap, level, n_levels)
            st.success("Summary tree created.")

        if st.button("Store Summary Embeddings"):
            with CapturePrints(log_callback=update_log):
                st.session_state.vectorstore = create_store_embeddings(st.session_state.clusters,st.session_state.total_pages, chunk_size, chunk_overlap, level, n_levels)
            st.success("Embeddings created and stored.")
            st.success('VectorStore is Loaded')

    else:
        with CapturePrints(log_callback=update_log):
            if 'vectorstore' not in st.session_state: 
                st.session_state.vectorstore = load_vectorstore("Pre_stored_Vec_Store", embd)
        st.success('Pre-stored VectorStore is Loaded')

    st.markdown('---')
    st.text_area("Log", st.session_state.log, height=300)


#################################################################################################################    
######################## Disable chat until API/LLM key is validated ############################################
#################################################################################################################



    
#################################################################################################################    
##################################   Chat Interface    ##########################################################    
# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if 'sessions' not in st.session_state:
    st.session_state.sessions = []

# Define utility functions
def save_session():
    if st.session_state.messages:
        st.session_state.sessions.append(st.session_state.messages.copy())

def new_session():
    save_session()
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.success("New session created and old session saved", icon='✅')

def display_generated_questions(questions):
    with st.expander("Generated Questions"):
        for i, q in enumerate(questions):
            st.write(f"Q{i+1}: {q}")

def display_retrieved_documents(retrieved_docs):
    with st.expander("Retrieved Documents"):
        for i, docs in enumerate(retrieved_docs):
            st.write(f"For Q{i+1}:")
            for doc in docs:
                content = doc.page_content.split('\n')[:2]  # Get the first 2 lines
                st.write(' '.join(content))

def display_final_context(final_context):
    with st.expander("Final Context"):
        st.write(final_context)

# Sidebar logic
with st.sidebar:
    col1, col2= st.columns([1,1])
    with col1:
        if st.button('Clear Log'):
            st.session_state.log = ""
    with col2:
        st.button('New Session', on_click=new_session)

col1, col2= st.columns([1,1])
with col1:
    if st.session_state.sessions:
        session_index = st.selectbox('Select Previous Session', range(len(st.session_state.sessions)), format_func=lambda x: f"Session {x+1}")
    if st.button('Load Selected Session'):
        st.session_state.messages = st.session_state.sessions[session_index].copy()
        st.success("Loaded selected session")
with col2:
    st.session_state.topk = st.number_input("Top K Retrieval", min_value=1, max_value=10, value=3, help='Number of chunks Retrieved')

# Main chat interface logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

disable_chat = not (llm_type == 'API' and api_key_valid or llm_type == 'Local' and selected_model)

if not disable_chat:
    # Capture the user's input
    
    if prompt := st.chat_input("Enter your question:", disabled=disable_chat):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Initialize the RAG chain if it doesn't exist
    if llm_type == 'Local':
        st.session_state.rag_chain = setup_ollama_language_model_chain(st.session_state.vectorstore, selected_model,topk=st.session_state.topk)
    else:
        st.session_state.rag_chain = setup_language_model_chain(st.session_state.vectorstore,topk=st.session_state.topk)
    
    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.topk})
                questions = generate_queries.invoke({"question": prompt})
                display_generated_questions(questions)
                
                retrieved_docs = [retriever.get_relevant_documents(q) for q in questions]
                display_retrieved_documents(retrieved_docs)
                
                unique_docs = get_unique_union(retrieved_docs)
                final_context = "\n\n".join([doc.page_content for doc in unique_docs])
                display_final_context(final_context)
                
                response = invoke_chain(st.session_state.rag_chain, prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
            
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
