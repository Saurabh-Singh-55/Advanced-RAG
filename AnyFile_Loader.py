from typing import Dict, List, Optional, Tuple
import glob
import os
from langchain.docstore.document import Document
from multiprocessing import Pool
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# Define custom loader mapping with file extensions and corresponding loader classes
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> List[Document]:
    """
    Load a single document based on its file extension, including file path in metadata.

    Parameters:
    - file_path: str, path to the file to be loaded.

    Returns:
    - List[Document]: A list of Document objects loaded from the file, with metadata including the file path.
    """
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()

        # Add file path to metadata of each Document
        for document in documents:
            document.metadata = {'file_path': file_path}
            # if 'metadata' in document.__dict__:  # Assuming the Document has a metadata attribute
            #     document.metadata['file_path'] = file_path
            # else:
            #     document.metadata = {'file_path': file_path}  # Create metadata if it doesn't exist

        return documents

    raise ValueError(f"Unsupported file extension '{ext}'")


    
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents from a specified directory, filtering out ignored files.

    Parameters:
    - source_dir: str, directory to load documents from.
    - ignored_files: List[str], list of file paths to ignore.

    Returns:
    - List[Document]: A list of Document objects loaded from the directory.
    """
    all_files = []
    for ext in LOADER_MAPPING.keys():
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))


    # Filter out ignored files
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    documents = []
    for file_path in filtered_files:
        try:
            docs = load_single_document(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading document from {file_path}: {e}")

    return documents
