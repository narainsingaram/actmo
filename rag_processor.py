import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Use updated import

# Define constants for model name and index path
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_FAISS_INDEX_PATH = 'faiss_index'
DEFAULT_DATA_PATH = 'data.txt'

def load_documents(data_path=DEFAULT_DATA_PATH):
    """Loads documents from the specified text file."""
    loader = TextLoader(data_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings(model_name=DEFAULT_EMBEDDING_MODEL):
    """Initializes and returns HuggingFace embeddings."""
    # Specify trust_remote_code=True if you are using a model that requires it
    # and you trust the source of the model.
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_and_save_faiss_index(texts, embeddings, index_path=DEFAULT_FAISS_INDEX_PATH):
    """Creates a FAISS index from texts and embeddings, and saves it locally."""
    if not texts:
        raise ValueError("No texts provided to create FAISS index. Ensure data.txt is not empty and processed correctly.")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index created and saved to {index_path}")

def load_faiss_index(index_path=DEFAULT_FAISS_INDEX_PATH, embeddings_model=DEFAULT_EMBEDDING_MODEL):
    """Loads a FAISS index from local storage."""
    if not os.path.exists(index_path):
        # If index doesn't exist, create it first
        print(f"FAISS index not found at {index_path}. Attempting to create it.")
        documents = load_documents()
        texts = split_documents(documents)
        embeddings_instance = create_embeddings(model_name=embeddings_model) # Renamed to avoid conflict
        if not texts:
            print("No documents found in data.txt. Cannot create FAISS index.")
            return None
        create_and_save_faiss_index(texts, embeddings_instance, index_path) # Pass the instance

    # Allow dangerous deserialization if you trust the source of the index file.
    # This is often required for FAISS.
    embeddings_instance_for_load = create_embeddings(model_name=embeddings_model) # Create a fresh instance for loading
    vectorstore = FAISS.load_local(index_path, embeddings_instance_for_load, allow_dangerous_deserialization=True)
    print(f"FAISS index loaded from {index_path}")
    return vectorstore

def get_relevant_documents(query, vectorstore, embeddings_model_name=DEFAULT_EMBEDDING_MODEL, k=3):
    """
    Performs a similarity search on the FAISS index and returns relevant documents.
    If vectorstore is None (e.g., index creation failed), returns an empty list.
    """
    if vectorstore is None:
        print("Vectorstore not available. Cannot retrieve relevant documents.")
        return []
    # The retriever should be created from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query) # Use invoke for LCEL compatibility
    return relevant_docs

if __name__ == '__main__':
    # This part is for testing the RAG processor independently
    # It will create the index if it doesn't exist.
    print("Running RAG processor test...")

    # 1. Load or create FAISS index
    # Ensure data.txt exists and has content before running this.
    # If data.txt was just created and is empty, this will cause an error in create_and_save_faiss_index.
    # For initial setup, it might be better to explicitly create the index first.

    # Check if data.txt exists and is not empty
    if not os.path.exists(DEFAULT_DATA_PATH) or os.path.getsize(DEFAULT_DATA_PATH) == 0:
        print(f"{DEFAULT_DATA_PATH} is missing or empty. Please create it with some text data first.")
        # Example: Create a dummy data.txt if it doesn't exist for the test run
        # with open(DEFAULT_DATA_PATH, "w") as f:
        #     f.write("This is some sample financial data for testing the RAG pipeline. " * 10)
        # print(f"Created a dummy {DEFAULT_DATA_PATH} for testing.")

    # Attempt to load the index (which will create it if it's not there)
    # This requires the embedding model to be downloaded on first run.
    try:
        # Ensure embeddings are created before loading/creating index
        embeddings = create_embeddings()

        # Check if index exists, if not, create it
        if not os.path.exists(DEFAULT_FAISS_INDEX_PATH):
            print("FAISS index not found. Creating new index...")
            documents = load_documents()
            if not documents:
                raise FileNotFoundError(f"No documents found in {DEFAULT_DATA_PATH}. Cannot create index.")
            texts = split_documents(documents)
            if not texts:
                raise ValueError("Splitting documents resulted in no texts. Cannot create index.")
            create_and_save_faiss_index(texts, embeddings, DEFAULT_FAISS_INDEX_PATH)
            print(f"Successfully created and saved FAISS index to {DEFAULT_FAISS_INDEX_PATH}")

        # Load the index
        vector_store = load_faiss_index(embeddings_model=DEFAULT_EMBEDDING_MODEL) # Pass model name

        if vector_store:
            # 2. Test document retrieval
            test_query = "What are call options?"
            print(f"\nTesting query: '{test_query}'")
            retrieved_docs = get_relevant_documents(test_query, vector_store)
            if retrieved_docs:
                print(f"\nFound {len(retrieved_docs)} relevant documents:")
                for i, doc in enumerate(retrieved_docs):
                    print(f"--- Document {i+1} ---")
                    print(doc.page_content) # Print the content of the document
                    print("--------------------")
            else:
                print("No relevant documents found for the test query.")
        else:
            print("Failed to load or create FAISS vector store. Retrieval test skipped.")

    except Exception as e:
        print(f"An error occurred during RAG processor test: {e}")
        print("Please ensure 'data.txt' exists and contains text, and that you have internet access for downloading models if it's the first run.")
