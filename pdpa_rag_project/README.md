# PDPA Legal Assistant (ผู้ช่วยตอบคำถามกฎหมาย PDPA)

แชตบอตตอบคำถามเกี่ยวกับพระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล (PDPA) ด้วยเทคนิค RAG (Retrieval-Augmented Generation)

## Domain & แนวคิด
ระบบช่วยค้นหาและตอบคำถามกฎหมาย PDPA เพื่ออำนวยความสะดวกให้ประชาชนและองค์กร สามารถตรวจสอบสิทธิ ข้อบังคับ และบทลงโทษตามกฎหมายได้อย่างถูกต้อง อ้างอิงตรงจากบทบัญญัติกฎหมาย

## เทคนิคทางวิทยาการข้อมูลที่ใช้ (RAG Pipeline)
1. **Document Loading & Chunking**: โหลดไฟล์เอกสารกฎหมาย (.txt) และแบ่งเป็น Chunk ขนาด 500 ตัวอักษร (Overlap 100)
2. **Embedding & Vector Search**: แปลงข้อความด้วยโมเดล `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` และจัดเก็บใน `FAISS Vector Database`
3. **Prompt Engineering**: บังคับให้ LLM ตอบเฉพาะจาก Context ที่ค้นเจอเท่านั้น หากไม่มีข้อมูลให้ปฏิเสธด้วยคำว่า "ไม่พบข้อมูลดังกล่าวในเอกสารความรู้"
4. **LLM Integration**: ประมวลผลคำตอบผ่าน Groq API (`llama-3.3-70b-versatile`)
5. **UI & Citation**: พัฒนาด้วย Streamlit แสดงประวัติการสนทนาพร้อมกดดู Source Documents อ้างอิงได้

## วิธีตั้งค่าเพื่อใช้งานแบบ Local
1. Clone Repository หรือ Extract Zip โครงสร้างไฟล์
2. ติดตั้ง Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. สร้างไฟล์ `.streamlit/secrets.toml` และระบุ API Key:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. รันแอปพลิเคชัน:
   ```bash
   streamlit run app.py
   ```
