# General utility and data handling
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import os
import glob

# UMAP for dimensionality reduction
import umap

# Gaussian Mixture Model for clustering
from sklearn.mixture import GaussianMixture

# LangChain and related components for embedding, summarization, and RAPTOR
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import (
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.openai import ChatOpenAI

# TikToken for token counting and related operations
import tiktoken

# ChromaDB for vector storage
from chromadb.config import Settings

# Plotting and visualization
import matplotlib.pyplot as plt

# LangChain Community and Hub components
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
