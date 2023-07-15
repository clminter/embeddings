import PyPDF4
import difflib
import streamlit as st
from sentence_transformers import SentenceTransformer

# Function to read PDF file and extract text
def read_pdf(file):
    text = ""
    pdf_reader = PyPDF4.PdfFileReader(file)
    for page in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page).extractText()
    return text

# Function to split text into chunks
def split_text(text, chunk_size):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to encode chunks using SentenceTransformer model
def encode_chunks(chunks):
    sentence_transformer_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = sentence_transformer_model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Function to compare chunks and generate a summary of changes
def generate_summary(chunks1, chunks2):
    summary = ""
    for chunk1, chunk2 in zip(chunks1, chunks2):
        diff = difflib.ndiff(chunk1.splitlines(), chunk2.splitlines())
        changes = [line[2:] for line in diff if line.startswith("+") or line.startswith("-")]
        if len(changes) == 0:
            summary += f"\nNo changes in this chunk.\n\n"
        else:
            summary += f"\nChanges in this chunk:\n"
            for change in changes:
                if change.startswith("+"):
                    summary += f"[{change}]\n"
                elif change.startswith("-"):
                    summary += f"[{change}]\n"
    return summary

# Main code
st.title("Semantic Search Question and Answer Application")

# Upload PDF files
pdf_file1 = st.file_uploader("Upload PDF File 1", type="pdf")
pdf_file2 = st.file_uploader("Upload PDF File 2", type="pdf")

if pdf_file1 and pdf_file2:
    # Read PDF files
    text1 = read_pdf(pdf_file1)
    text2 = read_pdf(pdf_file2)

    # Split text into chunks
    chunk_size = 500
    chunks1 = split_text(text1, chunk_size)
    chunks2 = split_text(text2, chunk_size)

    # Encode chunks using SentenceTransformer
    embeddings1 = encode_chunks(chunks1)
    embeddings2 = encode_chunks(chunks2)

    # Generate summary of changes
    summary = generate_summary(chunks1, chunks2)

    # Display changes and summary
    st.header("Changes")
    for i, (chunk1, chunk2) in enumerate(zip(chunks1, chunks2)):
        if chunk1 == chunk2:
            st.success(chunk1)
        else:
            st.error(chunk1)
            st.error(chunk2)

    st.sidebar.header("Summary of Changes")
    st.sidebar.text(summary)

    # Save summary to file
    with open("summary.txt", "w") as f:
        f.write(summary)

    st.success("Summary of changes saved to summary.txt.")
