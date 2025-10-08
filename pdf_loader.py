from PyPDF2 import PdfReader

def load_pdf(file_path):
    """Load and extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text