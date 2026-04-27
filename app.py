import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import gradio as gr
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5,
    )
    return llm

## Document loader
def document_loader(file):
    if file is None:
        return None
    file_path = file if isinstance(file, str) else file.name
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks

## Embedding model
def get_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Vector db
def vector_database(chunks):
    embedding_model = get_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## QA Chain Logic
def retriever_qa(file, query):
    if file is None:
        return "ERROR: INSERT DISK (PDF) TO CONTINUE..."
    if not query:
        return "ERROR: NO COMMAND ENTERED."
    
    try:
        llm = get_llm()
        splits = document_loader(file)
        chunks = text_splitter(splits)
        vectordb = vector_database(chunks)
        retriever_obj = vectordb.as_retriever()
        
        qa = RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type="stuff", 
                                        retriever=retriever_obj, 
                                        return_source_documents=False)
        response = qa.invoke(query)
        return response['result'].upper() # All caps for retro feel
    except Exception as e:
        return f"SYSTEM FAILURE: {str(e).upper()}"

# 8-BIT NEON RETRO CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

* {
    font-family: 'Press Start 2P', cursive !important;
    text-transform: uppercase;
}

body, .gradio-container {
    background-color: #000000 !important;
}

.container { 
    max-width: 1000px; 
    margin: auto; 
    padding: 20px;
    background: #000000;
    border: 6px solid #00ff00;
    box-shadow: 0 0 20px #00ff00;
}

.header { 
    text-align: center; 
    margin-bottom: 40px; 
    padding: 20px;
    border-bottom: 4px dashed #ff00ff;
}

.header h1 { 
    font-size: 2.5em; 
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
}

.header p { 
    color: #ff00ff; 
    font-size: 0.8em; 
    margin-top: 15px;
    text-shadow: 0 0 5px #ff00ff;
}

.gr-button-primary {
    background-color: #000000 !important;
    border: 4px solid #ffff00 !important;
    box-shadow: 0 0 10px #ffff00 !important;
    color: #ffff00 !important;
    font-size: 0.8em !important;
}

.gr-button-primary:hover {
    background-color: #ffff00 !important;
    color: #000 !important;
    box-shadow: 0 0 30px #ffff00 !important;
}

input, textarea, .gr-box, .gr-form {
    background-color: #000 !important;
    color: #00ff00 !important;
    border: 3px solid #00ffff !important;
    border-radius: 0px !important;
    box-shadow: inset 0 0 10px #00ffff !important;
}

label {
    color: #ff00ff !important;
    font-size: 0.7em !important;
    margin-bottom: 8px;
    text-shadow: 0 0 5px #ff00ff;
}

.footer { 
    text-align: center; 
    margin-top: 50px; 
    color: #00ff00; 
    font-size: 0.6em;
    text-shadow: 0 0 5px #00ff00;
}

/* Customizing the file upload area */
.file-preview {
    background: #000 !important;
    border: 2px solid #00ff00 !important;
}

footer { display: none !important; }
"""

# Build the UI with gr.Blocks
with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as rag_application:
    
    with gr.Column(elem_classes="container"):
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 👾 PDF BOT v1.0")
            gr.Markdown("INSERT PDF DATA AND ASK THE SYSTEM")
        
        with gr.Row():
            # Left Column: Terminal Input
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="[ DISK SLOT ]", 
                    file_count="single", 
                    file_types=['.pdf'], 
                    type="filepath"
                )
                gr.Markdown("---")
                gr.Markdown("### SYSTEM MANUAL")
                gr.Markdown("1. LOAD PDF\n2. INPUT CMD\n3. EXECUTE")
            
            # Right Column: Output Console
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="[ COMMAND LINE ]", 
                    placeholder="TYPE QUERY HERE...", 
                    lines=3
                )
                submit_btn = gr.Button("RUN EXECUTION", variant="primary")
                output_text = gr.Textbox(
                    label="[ SYSTEM OUTPUT ]", 
                    interactive=False, 
                    lines=10
                )
        
        # Action
        submit_btn.click(
            fn=retriever_qa,
            inputs=[file_input, query_input],
            outputs=output_text
        )

        # Footer
        with gr.Column(elem_classes="footer"):
            gr.Markdown("== COMPATIBLE WITH ALL RETRO BROWSERS ==")

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)
