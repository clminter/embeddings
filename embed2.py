import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from diff_match_patch import diff_match_patch

# Set up Streamlit layout
st.title("⏭️ Semantic Search Agent")
st.write("A MINTER 🤖")
st.subheader("Document Control: Change Highlighter")

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
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
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
    def semantic_search(query, documents, top_k=5, tree_depth=50, chunk_size=512):
        query_embedding = sentence_transformer_model.encode([query], convert_to_tensor=True)
        results = []
        for i, doc_text in enumerate(document_texts):
            chunks = [doc_text[i:i + chunk_size] for i in range(0, len(doc_text), chunk_size)]
            chunk_embeddings = sentence_transformer_model.encode(chunks, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
            top_results = torch.argsort(cos_scores, descending=True)[:top_k]
            results.extend([(i, result.item()) for result in top_results])
        return results

    # Display search input box
    search_query = st.text_input("Enter your search query")

    # Control tree depth and chunk size
    tree_depth = st.sidebar.slider("Tree Depth", min_value=1, max_value=100, value=50)
    chunk_size = st.sidebar.slider("Chunk Size", min_value=128, max_value=1024, step=128, value=512)

    # Perform semantic search and display results
    if st.button("Search"):
        if search_query.strip() == "":
            st.warning("Please enter a valid search query.")
        else:
            results = semantic_search(search_query, documents, top_k=5, tree_depth=tree_depth, chunk_size=chunk_size)
            st.subheader("Search Results")
            for i, (doc_index, result) in enumerate(results):
                if doc_index < len(document_names):
                    document_name = document_names[doc_index]
                    document_text = document_texts[doc_index]

                    st.write(f"**Result {i + 1}: {document_name}**")

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
                        st.write("Highlighted Changes:")
                        st.markdown(highlighted_text, unsafe_allow_html=True)

                        # Save summary of changes to a file
                        with open(f"{document_name}_summary.txt", "w") as summary_file:
                            summary_file.write(summary_text)
                        st.success(f"Summary of changes saved to {document_name}_summary.txt")

                    else:
                        st.write("No changes found.")

                    st.write("----------")
                else:
                    st.write(f"Result {i + 1} index exceeds the range of document_names.")

    # Display summary in sidebar
    if st.sidebar.checkbox("Show Summary of Changes"):
        summary_text = ""
        for doc_name in document_names:
            with open(f"{doc_name}_summary.txt", "r") as summary_file:
                summary_text += f"{doc_name}:\n{summary_file.read()}\n\n"
        st.sidebar.text_area("Summary", summary_text, height=400)

else:
    st.warning("Please upload PDF documents.")
