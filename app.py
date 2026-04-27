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
        return "!!! ERROR: NO INPUT DISK DETECTED !!!"
    if not query:
        return "!!! ERROR: COMMAND LINE EMPTY !!!"
    
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
        return response['result']
    except Exception as e:
        return f"!!! SYSTEM HALT: {str(e)} !!!"

# ANSI ART HEADER (HTML/CSS SIMULATION)
ansi_art = """
<pre style='line-height: 1.1; font-family: monospace; font-weight: bold; background: #000; padding: 20px; border: 4px double #555; text-align: center; color: #fff;'>
<span style='color: #ff5555;'>  _____  </span><span style='color: #55ff55;'>_____  </span><span style='color: #5555ff;'>______ </span>  <span style='color: #ffff55;'>____  </span> <span style='color: #ff55ff;'> ____ </span><span style='color: #55ffff;'>_______ </span>
<span style='color: #ff5555;'> |  __ \\</span><span style='color: #55ff55;'>|  __ \\</span><span style='color: #5555ff;'>|  ____|</span> <span style='color: #ffff55;'>|  _ \\ </span><span style='color: #ff55ff;'>/ __ \\</span><span style='color: #55ffff;'>__   __|</span>
<span style='color: #ff5555;'> | |__) </span><span style='color: #55ff55;'>| |  | |</span><span style='color: #5555ff;'> |__   </span> <span style='color: #ffff55;'>| |_) |</span><span style='color: #ff55ff;'>| |  | |</span><span style='color: #55ffff;'>  | |   </span>
<span style='color: #ff5555;'> |  ___/</span><span style='color: #55ff55;'>| |  | |</span><span style='color: #5555ff;'>  __|  </span> <span style='color: #ffff55;'>|  _ < </span><span style='color: #ff55ff;'>| |  | |</span><span style='color: #55ffff;'>  | |   </span>
<span style='color: #ff5555;'> | |    </span><span style='color: #55ff55;'>| |__| |</span><span style='color: #5555ff;'> |     </span> <span style='color: #ffff55;'>| |_) |</span><span style='color: #ff55ff;'>| |__| |</span><span style='color: #55ffff;'>  | |   </span>
<span style='color: #ff5555;'> |_|    </span><span style='color: #55ff55;'>|_____/</span><span style='color: #5555ff;'>|_|     </span> <span style='color: #ffff55;'>|____/ </span><span style='color: #ff55ff;'>\\____/ </span><span style='color: #55ffff;'> |_|   </span>
<br>
<span style='color: #aaa;'>[ VERSION 2.0 - BBS EDITION ]</span>
</pre>
"""

# CLASSIC BBS / ANSI CSS
custom_css = """
body, .gradio-container {
    background-color: #0000aa !important; /* Classic BBS Blue */
}

.container { 
    max-width: 950px; 
    margin: auto; 
    padding: 20px;
    background: #000000;
    border: 5px solid #aaaaaa;
    box-shadow: 15px 15px 0px #000;
}

* {
    font-family: 'Courier New', Courier, monospace !important;
}

label {
    background: #aaaaaa !important;
    color: #000000 !important;
    padding: 2px 10px !important;
    font-weight: bold !important;
    text-transform: uppercase;
}

input, textarea, .gr-box, .gr-form {
    background-color: #000 !important;
    color: #ffffff !important;
    border: 2px solid #555555 !important;
    border-radius: 0px !important;
}

.gr-button-primary {
    background-color: #00aaaa !important;
    color: #ffffff !important;
    border: 2px solid #ffffff !important;
    border-radius: 0px !important;
    font-weight: bold !important;
    text-transform: uppercase;
}

.gr-button-primary:hover {
    background-color: #ffffff !important;
    color: #00aaaa !important;
}

.footer { 
    text-align: center; 
    margin-top: 30px; 
    color: #aaaaaa; 
    font-weight: bold;
}

footer { display: none !important; }
"""

with gr.Blocks(css=custom_css) as rag_application:
    
    with gr.Column(elem_classes="container"):
        # ANSI Art Header
        gr.HTML(ansi_art)
        
        with gr.Row():
            # Side Panel
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="FILE UPLOAD", 
                    file_count="single", 
                    file_types=['.pdf'], 
                    type="filepath"
                )
                gr.Markdown("---")
                gr.Markdown("### SYSTEM INFO")
                gr.Markdown("- **STATUS:** ONLINE\n- **CPU:** 80386\n- **BAUD:** 14400")
            
            # Main Console
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="PROMPT", 
                    placeholder="ENTER QUERY...", 
                    lines=3
                )
                submit_btn = gr.Button("EXECUTE COMMAND", variant="primary")
                output_text = gr.Textbox(
                    label="SYSTEM RESPONSE", 
                    interactive=False, 
                    lines=12
                )
        
        # Action
        submit_btn.click(
            fn=retriever_qa,
            inputs=[file_input, query_input],
            outputs=output_text
        )

        # Footer
        with gr.Column(elem_classes="footer"):
            gr.Markdown("(C) 1992 PDF-BOT SYSTEMS UNLIMITED")

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)
