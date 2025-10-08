# website_loader.py
from langchain_community.document_loaders import WikipediaLoader

def load_website(page_title: str, lang: str = "en") -> str:
    """Load a Wikipedia page by title"""
    loader = WikipediaLoader(
        query=page_title,
        lang=lang,
        load_max_docs=1,
        doc_content_chars_max=10000  # Adjust as needed
    )
    
    documents = loader.load()
    if documents:
        return documents[0].page_content
    else:
        return "No content found for the given page title."

# Usage example:
# website_text = load_wikipedia_page("Artificial intelligence")