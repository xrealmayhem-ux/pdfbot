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


# ╔══════════════════════════════════════════╗
#   CYBERPUNK ANSI ART HEADER
# ╚══════════════════════════════════════════╝
ansi_art = """
<div class="cyber-header">
  <div class="scanlines"></div>
  <div class="corner tl">┌─</div>
  <div class="corner tr">─┐</div>
  <div class="corner bl">└─</div>
  <div class="corner br">─┘</div>
  <pre class="ascii-title">
<span class="c1">██████╗ </span><span class="c2">██████╗ </span><span class="c3">███████╗</span>  <span class="c4">██████╗  </span><span class="c5">██████╗ </span><span class="c6">████████╗</span>
<span class="c1">██╔══██╗</span><span class="c2">██╔══██╗</span><span class="c3">██╔════╝</span>  <span class="c4">██╔══██╗</span><span class="c5">██╔═══██╗</span><span class="c6">╚══██╔══╝</span>
<span class="c1">██████╔╝</span><span class="c2">██║  ██║</span><span class="c3">█████╗  </span>  <span class="c4">██████╔╝</span><span class="c5">██║   ██║</span><span class="c6">   ██║   </span>
<span class="c1">██╔═══╝ </span><span class="c2">██║  ██║</span><span class="c3">██╔══╝  </span>  <span class="c4">██╔══██╗</span><span class="c5">██║   ██║</span><span class="c6">   ██║   </span>
<span class="c1">██║     </span><span class="c2">██████╔╝</span><span class="c3">██║     </span>  <span class="c4">██████╔╝</span><span class="c5">╚██████╔╝</span><span class="c6">   ██║   </span>
<span class="c1">╚═╝     </span><span class="c2">╚═════╝ </span><span class="c3">╚═╝     </span>  <span class="c4">╚═════╝ </span><span class="c5"> ╚═════╝ </span><span class="c6">   ╚═╝   </span></pre>
  <div class="tagline">
    <span class="blink">▶</span>
    NEURAL DOCUMENT INTERFACE &nbsp;/&nbsp; v2.0 CYBER EDITION
    <span class="blink">◀</span>
  </div>
  <div class="status-bar">
    <span class="stat online">● SYSTEM ONLINE</span>
    <span class="stat">◈ MODEL: MISTRAL-7B</span>
    <span class="stat">◈ EMBED: MiniLM-L6</span>
    <span class="stat pulse">◈ AWAITING INPUT</span>
  </div>
</div>
"""


# ╔══════════════════════════════════════════╗
#   CYBERPUNK CSS
# ╚══════════════════════════════════════════╝
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

:root {
  --cyan:    #00cc88;
  --magenta: #ffb000;
  --yellow:  #ccff00;
  --green:   #00ff5f;
  --bg:      #0c0c0c;
  --bg2:     #111111;
  --border:  #1f2f1f;
  --text:    #b0d0b0;
  --dim:     #3a5a3a;
}

*, *::before, *::after {
  box-sizing: border-box;
  font-family: 'Share Tech Mono', monospace !important;
}

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  min-height: 100vh;
}

/* ── HEADER ── */
.cyber-header {
  position: relative;
  padding: 28px 20px 16px;
  background: linear-gradient(180deg, #0a1a0a 0%, #0c0c0c 100%);
  border: 1px solid #00cc88;
  box-shadow: 0 0 30px #00cc8818, inset 0 0 60px #00ff0008;
  overflow: hidden;
  margin-bottom: 4px;
}

.scanlines {
  position: absolute; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.15) 2px,
    rgba(0,0,0,0.15) 4px
  );
  pointer-events: none;
  z-index: 1;
}

.corner {
  position: absolute;
  font-size: 14px;
  color: var(--cyan);
  text-shadow: 0 0 8px var(--cyan);
  z-index: 2;
}
.tl { top: 6px;  left: 8px; }
.tr { top: 6px;  right: 8px; }
.bl { bottom: 6px; left: 8px; }
.br { bottom: 6px; right: 8px; }

.ascii-title {
  font-size: clamp(5px, 1.1vw, 11px);
  line-height: 1.15;
  text-align: center;
  margin: 0 0 10px;
  letter-spacing: 0;
  position: relative;
  z-index: 2;
}

.c1 { color: #00ff5f; text-shadow: 0 0 10px #00ff5f88; }
.c2 { color: #00cc88; text-shadow: 0 0 10px #00cc8888; }
.c3 { color: #ccff00; text-shadow: 0 0 10px #ccff0088; }
.c4 { color: #00ff5f; text-shadow: 0 0 10px #00ff5f88; }
.c5 { color: #00cc88; text-shadow: 0 0 10px #00cc8888; }
.c6 { color: #ffb000; text-shadow: 0 0 10px #ffb00088; }

.tagline {
  text-align: center;
  color: var(--cyan);
  font-size: 11px;
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-bottom: 12px;
  text-shadow: 0 0 10px var(--cyan);
  position: relative;
  z-index: 2;
}

.status-bar {
  display: flex;
  justify-content: center;
  gap: 24px;
  font-size: 10px;
  letter-spacing: 2px;
  flex-wrap: wrap;
  position: relative;
  z-index: 2;
}

.stat        { color: var(--dim); }
.stat.online { color: var(--green); text-shadow: 0 0 8px var(--green); }
.stat.pulse  { color: var(--yellow); text-shadow: 0 0 8px var(--yellow); animation: flicker 3s infinite; }

@keyframes flicker {
  0%,95%,100% { opacity: 1; }
  96%          { opacity: 0.3; }
  97%          { opacity: 1; }
  98%          { opacity: 0.5; }
}

@keyframes blink {
  0%,49%  { opacity: 1; }
  50%,100% { opacity: 0; }
}

.blink { animation: blink 1s step-end infinite; }

/* ── MAIN CONTAINER ── */
.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
  background: var(--bg2);
  border: 1px solid var(--border);
}

/* ── PANEL TITLES ── */
.panel-label {
  font-family: 'Orbitron', monospace !important;
  font-size: 9px;
  letter-spacing: 3px;
  color: var(--cyan);
  text-transform: uppercase;
  border-bottom: 1px solid var(--cyan);
  padding-bottom: 4px;
  margin-bottom: 12px;
  text-shadow: 0 0 8px var(--cyan);
}

/* ── LABELS ── */
label, .gr-block label, span.svelte-1gfkfd6 {
  color: var(--cyan) !important;
  font-size: 10px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  background: transparent !important;
  padding: 0 !important;
  font-weight: normal !important;
  text-shadow: 0 0 6px var(--cyan) !important;
}

/* ── INPUTS ── */
input, textarea,
.gr-box, .gr-form,
[data-testid="textbox"] textarea,
.gr-input, .gr-text-input {
  background: #050515 !important;
  color: var(--green) !important;
  border: 1px solid #1a3a3a !important;
  border-radius: 0 !important;
  caret-color: var(--cyan) !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}

input:focus, textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 12px #00ffe720, inset 0 0 20px #00ffe708 !important;
  outline: none !important;
}

/* ── FILE UPLOAD ── */
.gr-file, [data-testid="file"] {
  background: #050515 !important;
  border: 1px dashed #1a4a4a !important;
  border-radius: 0 !important;
  color: var(--text) !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.gr-file:hover {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 16px #00ffe718 !important;
}

/* ── BUTTON ── */
.gr-button-primary, button.primary {
  background: transparent !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan) !important;
  border-radius: 0 !important;
  font-size: 11px !important;
  letter-spacing: 4px !important;
  text-transform: uppercase !important;
  font-family: 'Orbitron', monospace !important;
  padding: 12px 24px !important;
  position: relative;
  overflow: hidden;
  transition: all 0.2s;
  text-shadow: 0 0 8px var(--cyan) !important;
  box-shadow: 0 0 16px #00ffe718, inset 0 0 16px #00ffe705 !important;
}

.gr-button-primary:hover, button.primary:hover {
  background: var(--cyan) !important;
  color: #000 !important;
  box-shadow: 0 0 30px #00ffe760, inset 0 0 20px #00ffe720 !important;
  text-shadow: none !important;
}

/* ── OUTPUT BOX ── */
[data-testid="textbox"][readonly] textarea,
.output-text textarea {
  background: #020210 !important;
  color: var(--green) !important;
  border: 1px solid #0a2a0a !important;
  border-left: 2px solid var(--green) !important;
  box-shadow: inset 0 0 30px #00ff0808 !important;
}

/* ── SYSTEM INFO ── */
.sysinfo {
  border: 1px solid #1a1a3a;
  padding: 12px;
  font-size: 11px;
  color: var(--dim);
  line-height: 2;
}

.sysinfo .key   { color: #ffb000; text-shadow: 0 0 6px #ffb00066; }
.sysinfo .val   { color: var(--text); }
.sysinfo .ok    { color: var(--green); }
.sysinfo .warn  { color: var(--yellow); animation: flicker 4s infinite; }

/* ── DIVIDER ── */
.cyber-divider {
  border: none;
  border-top: 1px solid #1a1a3a;
  margin: 16px 0;
  position: relative;
}

.cyber-divider::after {
  content: '◈';
  position: absolute;
  top: -8px; left: 50%;
  transform: translateX(-50%);
  background: var(--bg2);
  padding: 0 8px;
  color: var(--dim);
  font-size: 10px;
}

/* ── FOOTER ── */
.cyber-footer {
  text-align: center;
  margin-top: 20px;
  padding: 12px;
  border-top: 1px solid #1a1a3a;
  font-size: 10px;
  color: var(--dim);
  letter-spacing: 3px;
}

.cyber-footer span { color: #ffb000; text-shadow: 0 0 6px #ffb00066; }

footer { display: none !important; }
"""


# ╔══════════════════════════════════════════╗
#   GRADIO UI
# ╚══════════════════════════════════════════╝
with gr.Blocks(css=custom_css) as rag_application:

    with gr.Column(elem_classes="container"):

        # Header
        gr.HTML(ansi_art)

        with gr.Row():

            # ── LEFT PANEL ──
            with gr.Column(scale=1):
                gr.HTML("<div class='panel-label'>◈ Input Node</div>")
                file_input = gr.File(
                    label="Load Document",
                    file_count="single",
                    file_types=['.pdf'],
                    type="filepath"
                )
                gr.HTML("<hr class='cyber-divider'>")
                gr.HTML("""
                <div class='panel-label'>◈ System Status</div>
                <div class='sysinfo'>
                  <div><span class='key'>STATUS  </span> <span class='val ok'>● ONLINE</span></div>
                  <div><span class='key'>LLM     </span> <span class='val'>Mistral-7B-v0.3</span></div>
                  <div><span class='key'>EMBED   </span> <span class='val'>MiniLM-L6-v2</span></div>
                  <div><span class='key'>VDB     </span> <span class='val'>ChromaDB</span></div>
                  <div><span class='key'>CHUNKS  </span> <span class='val'>1000 / 100</span></div>
                  <div><span class='key'>TOKENS  </span> <span class='val warn'>512 MAX</span></div>
                </div>
                """)

            # ── RIGHT PANEL ──
            with gr.Column(scale=2):
                gr.HTML("<div class='panel-label'>◈ Query Interface</div>")
                query_input = gr.Textbox(
                    label="Neural Prompt",
                    placeholder="> ENTER QUERY..._",
                    lines=3
                )
                submit_btn = gr.Button("⟫ Execute Query", variant="primary")
                gr.HTML("<div style='height:8px'></div>")
                output_text = gr.Textbox(
                    label="System Response",
                    interactive=False,
                    lines=12,
                    elem_classes="output-text"
                )

        # Footer
        gr.HTML("""
        <div class='cyber-footer'>
          <span>PDF-BOT SYSTEMS</span> &nbsp;/&nbsp; NEURAL INTERFACE v2.0
          &nbsp;/&nbsp; ALL QUERIES PROCESSED LOCALLY
        </div>
        """)

    submit_btn.click(
        fn=retriever_qa,
        inputs=[file_input, query_input],
        outputs=output_text
    )

if __name__ == "__main__":
    rag_application.launch(server_name="[IP_ADDRESS]", server_port=7860)