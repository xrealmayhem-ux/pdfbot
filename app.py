import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import gradio as gr
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM via InferenceClient
def query_llm(context, question):
    client = InferenceClient(
        token=os.environ.get("HF_TOKEN"),
    )
    messages = [
        {
            "role": "user",
            "content": f"Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]
    response = client.chat_completion(
        model="google/gemma-2-2b-it",
        messages=messages,
        max_tokens=512,
        temperature=0.5
    )
    return response.choices[0].message.content

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
        splits = document_loader(file)
        chunks = text_splitter(splits)
        vectordb = vector_database(chunks)
        retriever_obj = vectordb.as_retriever()

        docs = retriever_obj.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        return query_llm(context, query)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"!!! SYSTEM HALT: {str(e)} !!!"


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
    <span class="stat">◈ MODEL: Mixtral-8x7B</span>
    <span class="stat">◈ EMBED: MiniLM-L6</span>
    <span class="stat pulse">◈ AWAITING INPUT</span>
  </div>
</div>
"""

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --accent:   #5e6ad2;
  --accent2:  #26b5a0;
  --danger:   #e05d5d;
  --warn:     #e0a030;
  --ok:       #3ecf8e;
  --bg:       #0d0d12;
  --surface:  rgba(255,255,255,0.04);
  --surface2: rgba(255,255,255,0.07);
  --border:   rgba(255,255,255,0.08);
  --border2:  rgba(255,255,255,0.14);
  --text:     #e2e4f0;
  --muted:    #6b7280;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Inter', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  min-height: 100vh;
  font-family: var(--sans) !important;
}

.cyber-header {
  position: relative;
  padding: 32px 24px 20px;
  background: var(--surface);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--border2);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 12px;
}

.cyber-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(94,106,210,0.08) 0%, transparent 60%, rgba(38,181,160,0.06) 100%);
  pointer-events: none;
}

.scanlines { display: none; }
.corner { display: none; }

.ascii-title {
  font-family: var(--mono) !important;
  font-size: clamp(5px, 1.05vw, 10px);
  line-height: 1.2;
  text-align: center;
  margin: 0 0 16px;
  letter-spacing: 0;
  position: relative;
  z-index: 2;
}

.c1 { color: #5e6ad2; }
.c2 { color: #7c86e0; }
.c3 { color: #26b5a0; }
.c4 { color: #3ecf8e; }
.c5 { color: #26b5a0; }
.c6 { color: #5e6ad2; }

.tagline {
  text-align: center;
  color: var(--muted);
  font-family: var(--sans) !important;
  font-size: 11px;
  font-weight: 400;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 16px;
  position: relative;
  z-index: 2;
}

.tagline .blink { color: var(--accent); }

.status-bar {
  display: flex;
  justify-content: center;
  gap: 6px;
  font-family: var(--mono) !important;
  font-size: 10px;
  flex-wrap: wrap;
  position: relative;
  z-index: 2;
}

.stat {
  color: var(--muted);
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 10px;
  letter-spacing: 1px;
}

.stat.online { color: var(--ok);   border-color: rgba(62,207,142,0.3); background: rgba(62,207,142,0.08); }
.stat.pulse  { color: var(--warn); border-color: rgba(224,160,48,0.3);  background: rgba(224,160,48,0.08); animation: flicker 4s infinite; }

@keyframes flicker {
  0%,94%,100% { opacity: 1; }
  95%          { opacity: 0.4; }
  97%          { opacity: 1; }
  99%          { opacity: 0.6; }
}

@keyframes blink {
  0%,49%   { opacity: 1; }
  50%,100% { opacity: 0; }
}

.blink { animation: blink 1.2s step-end infinite; }

.container {
  max-width: 1020px;
  margin: 0 auto;
  padding: 20px;
}

.panel-label {
  font-family: var(--sans) !important;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 2px;
  color: var(--muted);
  text-transform: uppercase;
  padding-bottom: 8px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

label, .gr-block label, span.svelte-1gfkfd6 {
  font-family: var(--sans) !important;
  color: var(--muted) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  background: transparent !important;
  padding: 0 !important;
  text-shadow: none !important;
}

input, textarea,
.gr-box, .gr-form,
[data-testid="textbox"] textarea,
.gr-input, .gr-text-input {
  font-family: var(--mono) !important;
  background: var(--surface) !important;
  backdrop-filter: blur(12px) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  transition: border-color 0.15s, background 0.15s;
}

input:focus, textarea:focus {
  border-color: var(--accent) !important;
  background: rgba(94,106,210,0.06) !important;
  outline: none !important;
  box-shadow: none !important;
}

.gr-file, [data-testid="file"] {
  background: var(--surface) !important;
  backdrop-filter: blur(12px) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 8px !important;
  color: var(--muted) !important;
  transition: border-color 0.15s, background 0.15s;
}

.gr-file:hover {
  border-color: var(--accent) !important;
  background: rgba(94,106,210,0.06) !important;
}

.gr-button-primary, button.primary {
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 24px !important;
  transition: all 0.15s;
  box-shadow: none !important;
  text-shadow: none !important;
}

.gr-button-primary:hover, button.primary:hover {
  background: #6e78e0 !important;
  transform: translateY(-1px);
}

[data-testid="textbox"][readonly] textarea,
.output-text textarea {
  font-family: var(--mono) !important;
  background: var(--surface) !important;
  backdrop-filter: blur(16px) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-left: 2px solid var(--accent2) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}

.sysinfo {
  background: var(--surface);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  font-family: var(--mono) !important;
  font-size: 11px;
  line-height: 2;
}

.sysinfo .key  { color: var(--muted); }
.sysinfo .val  { color: var(--text); }
.sysinfo .ok   { color: var(--ok); }
.sysinfo .warn { color: var(--warn); }

.cyber-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 16px 0;
  position: relative;
}

.cyber-divider::after {
  content: '◈';
  position: absolute;
  top: -8px; left: 50%;
  transform: translateX(-50%);
  background: var(--bg);
  padding: 0 8px;
  color: var(--muted);
  font-size: 10px;
}

.cyber-footer {
  text-align: center;
  margin-top: 20px;
  padding: 14px;
  border-top: 1px solid var(--border);
  font-family: var(--sans) !important;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 2px;
  color: var(--muted);
  text-transform: uppercase;
}

.cyber-footer span { color: var(--accent); }

footer { display: none !important; }
"""

with gr.Blocks(css=custom_css) as rag_application:

    with gr.Column(elem_classes="container"):

        gr.HTML(ansi_art)

        with gr.Row():

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
                  <div><span class='key'>LLM     </span> <span class='val'>Mixtral-8x7B</span></div>
                  <div><span class='key'>EMBED   </span> <span class='val'>MiniLM-L6-v2</span></div>
                  <div><span class='key'>VDB     </span> <span class='val'>ChromaDB</span></div>
                  <div><span class='key'>CHUNKS  </span> <span class='val'>1000 / 100</span></div>
                  <div><span class='key'>TOKENS  </span> <span class='val warn'>512 MAX</span></div>
                </div>
                """)

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
    rag_application.launch(server_name="0.0.0.0", server_port=7860)