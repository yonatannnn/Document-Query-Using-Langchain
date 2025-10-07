from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import cassio
import os
from dotenv import load_dotenv

load_dotenv()

def load_pdf(file_path):
    """Load and extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def setup_cassandra_vectorstore():
    """Initialize Cassandra vector store"""
    # Initialize cassio
    cassio.init(
        token=os.getenv("ASTRA_DB_APP_TOKEN"),
        database_id=os.getenv("ASTRA_DB_ID")
    )
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    vector_store = Cassandra(
        embedding=embeddings,
        table_name="document_qa",
        session=None,
        keyspace=None
    )
    
    return vector_store

if __name__ == "__main__":
    # Check if all environment variables are set
    required_vars = ["ASTRA_DB_ID", "ASTRA_DB_APP_TOKEN", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"Error: {var} is not set in environment variables")
            exit(1)
    
    # Load PDF
    pdf_text = load_pdf("soccer.pdf")
    print("PDF loaded successfully")
    print(f"First 500 characters: {pdf_text[:500]}...")
    
    # Initialize vector store
    print("Setting up Cassandra vector store...")
    vector_store = setup_cassandra_vectorstore()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200

    )
    texts = text_splitter.split_text(pdf_text)
    print(f"Split into {len(texts)} chunks")
    
    # Add to vector store
    print("Adding documents to vector store...")
    vector_store.add_texts(texts)
    
    # Create index wrapper
    index = VectorStoreIndexWrapper(vectorstore=vector_store)
    
    # Test query
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    
    query = input("Enter your question: ")
    answer = index.query(query, llm=llm)
    print(f"Q: {query}")
    print(f"A: {answer}")