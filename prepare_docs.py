import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.rag import RagSystem

def build_brain(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file {pdf_path}")
        return

    print(f"Đang đọc tài liệu: {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # Băm nhỏ tài liệu (mỗi đoạn ~500 ký tự như bản đặc tả)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    print(f"Đã băm thành {len(split_docs)} đoạn văn bản.")

    # Khởi tạo RAG và tạo Index
    rag = RagSystem()
    rag.create_index_from_documents(split_docs, save_path="faiss_index")
    print("Xây dựng não bộ FAISS thành công! Bạn có thể chạy server bây giờ.")

if __name__ == "__main__":
    # Để file tài liệu (PDF) của bạn ở đây
    PDF_FILE = "tai_lieu_tuyen_sinh_100_trang.pdf" 
    build_brain(PDF_FILE)
