import os
import glob
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="ผู้ช่วยกฎหมาย PDPA (PDPA Legal Assistant)",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ ผู้ช่วยตอบคำถามกฎหมาย PDPA (RAG System)")
st.caption("ระบบตอบคำถามกฎหมายคุ้มครองข้อมูลส่วนบุคคล อ้างอิงจากคลังเอกสารกฎหมาย")

# 2. Get API Key from Streamlit Secrets or Environment Variables
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ ไม่พบ GROQ_API_KEY กรุณาตั้งค่าใน Streamlit Secrets (`.streamlit/secrets.toml`)")
    st.info("ตัวอย่างใน Secrets: GROQ_API_KEY = \"gsk_...\"")
    st.stop()

client = Groq(api_key=api_key)

# 3. Cache Vector Database Loading
@st.cache_resource
def load_vector_store():
    data_folder = "data"
    file_paths = glob.glob(os.path.join(data_folder, "*.txt"))
    
    if not file_paths:
        st.error("ไม่พบไฟล์เอกสารในโฟลเดอร์ data/")
        return None
        
    documents = []
    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        documents.extend(docs)

    # Document Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Sentence Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Vector Database Creation
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

with st.spinner("กำลังโหลดคลังข้อมูลกฎหมาย... (ดำเนินการครั้งแรกเท่านั้น)"):
    vectorstore = load_vector_store()

# 4. Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ มีข้อสงสัยเกี่ยวกับกฎหมาย PDPA สอบถามได้เลยครับ"}
    ]

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 เอกสารอ้างอิงที่ใช้"):
                for src in msg["sources"]:
                    st.write(f"- **{src['source']}**: {src['content']}")

# 5. User Input Handling & RAG Processing
if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับ PDPA ที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและประมวลผลคำตอบ..."):
            docs_and_scores = vectorstore.similarity_search_with_score(prompt, k=3)
            
            context_text = ""
            sources = []
            for doc, score in docs_and_scores:
                src_name = doc.metadata.get("source", "ไม่ระบุ")
                context_text += f"\n[แหล่งที่มา: {src_name}]\n{doc.page_content}\n"
                sources.append({"source": src_name, "content": doc.page_content})

            system_prompt = f"""คุณคือผู้ช่วยตอบคำถามกฎหมาย PDPA มืออาชีพ
หน้าที่ของคุณคือตอบคำถามโดยอิงจากบริบท (Context) ที่กำหนดให้เท่านั้น

กฎข้อบังคับอย่างเคร่งครัด:
1. ตอบคำถามด้วยภาษาไทยที่เป็นทางการ เข้าใจง่าย
2. ตอบคำถามโดยใช้อ้างอิงจาก Context ด้านล่างนี้เท่านั้น
3. หากใน Context ไม่มีข้อมูลที่ใช้ตอบคำถาม ให้ตอบว่า "ไม่พบข้อมูลดังกล่าวในเอกสารความรู้" ห้ามเดาหรือสร้างข้อมูลขึ้นเองเด็ดขาด

Context ที่ค้นพบ:
{context_text}
"""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            st.markdown(answer)
            if sources:
                with st.expander("📚 เอกสารอ้างอิงที่ใช้"):
                    for src in sources:
                        st.write(f"- **{src['source']}**: {src['content']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
