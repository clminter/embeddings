import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import pdfplumber
from diff_match_patch import diff_match_patch

# Set up Streamlit layout
st.title(" ⏭️Semantic Search Question and Answer Agent")
st.write("A MINTER 🤖")
st.subheader("Query Multiple PDF Documents")

# Select and load pre-trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load SentenceTransformer model for embeddings
sentence_transformer_model = SentenceTransformer(model_name)

# Select and load PDF documents
pdf_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

# Read and preprocess PDF documents
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

documents = []
if pdf_files:
    for pdf_file in pdf_files:
        text = read_pdf(pdf_file)
        documents.append((pdf_file.name, text))

# Encode document embeddings
if documents:
    document_names = [doc[0] for doc in documents]
    document_texts = [doc[1] for doc in documents]
    document_embeddings = sentence_transformer_model.encode(document_texts, convert_to_tensor=True)

    # Semantic search function
    @st.cache_data
    def semantic_search(query, documents, top_k=5):
        query_embedding = sentence_transformer_model.encode([query], convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        top_results = torch.argsort(cos_scores, descending=True)[:top_k]
        return top_results

    # Display search input box
    search_query = st.text_input("Enter your search query")

    # Perform semantic search and display results
    if st.button("Search"):
        if search_query.strip() == "":
            st.warning("Please enter a valid search query.")
        else:
            results = semantic_search(search_query, documents, top_k=5)
            st.subheader("Search Results")
            for i, result in enumerate(results):
                document_name = document_names[result.item()]
                document_text = document_texts[result.item()]

                st.write(f"**Result {i+1}: {document_name}**")

                diffs = diff_match_patch().diff_main(document_text, search_query)
                highlighted_text = ""
                summary_text = ""
                for diff in diffs:
                    if diff[0] == -1:  # Deleted text
                        highlighted_text += "<span style='background-color: #FFCCCC'>{}</span>".format(diff[1])
                        summary_text += f"[Deleted]: {diff[1]}\n"
                    elif diff[0] == 1:  # Inserted text
                        highlighted_text += "<span style='background-color: #CCFFCC'>{}</span>".format(diff[1])
                        summary_text += f"[Inserted]: {diff[1]}\n"

                if highlighted_text:
                    st.write(f"Highlighted Changes:")
                    st.markdown(highlighted_text, unsafe_allow_html=True)

                    # Save summary of changes to a file
                    with open(f"{document_name}_summary.txt", "w") as summary_file:
                        summary_file.write(summary_text)
                    st.success(f"Summary of changes saved to {document_name}_summary.txt")

                    # Display summary in a separate column
                    st.write("Summary of Changes")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Summary", summary_text, height=400)
                    with col2:
                        st.write("You can download the summary file from the link below:")
                        st.markdown(f"[Download {document_name}_summary.txt](./{document_name}_summary.txt)")
                else:
                    st.write("No changes found.")