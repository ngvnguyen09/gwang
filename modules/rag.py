from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import openai
import os

logger = logging.getLogger("rag_module")

# The user mentioned 'keepitreal/vietnamese-sbert'
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
LLAMA_API_BASE = "http://localhost:11434/v1"
LLAMA_API_KEY = "llama.cpp" # Usually arbitrary for local server

SYSTEM_PROMPT = """Bạn là NCT-erBot, học sinh xuất sắc và đại sứ AI của trường THPT Nguyễn Công Trứ.
Nhiệm vụ: Trả lời ngắn gọn, tự nhiên bằng giọng điệu dễ thương, năng lượng dựa VÀO DUY NHẤT tài liệu truy xuất. 
Nếu câu hỏi KHÔNG có thông tin trong tài liệu, đáp: 'Dạ, vùng thông tin này robot chưa được nạp, bạn có thể hỏi khu vực Tư vấn để biết chi tiết nha!'.
Tuyệt đối KHÔNG BỊA RA thông tin ngoài tài liệu.
"""

class RagSystem:
    def __init__(self, index_path: str = None):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = None
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def load_index(self, path: str):
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")

    def create_index_from_documents(self, docs, save_path: str = "faiss_index"):
        logger.info("Creating new FAISS index from documents...")
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(save_path)
        logger.info(f"FAISS index saved to {save_path}")

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return "No knowledge base loaded."
            
        logger.info(f"Retrieving for query: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

def generate_response_stream(query: str, context: str):
    """
    Generator that yields tokens from Llama.cpp API in streaming fashion
    """
    client = openai.OpenAI(base_url=LLAMA_API_BASE, api_key=LLAMA_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tài liệu thông tin:\n{context}\n\nCâu hỏi: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama-3", # Ensure this matches your running model name in llama.cpp
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        yield "Đã có lỗi kết nối tới não bộ cục bộ LLM."
