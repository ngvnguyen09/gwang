import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from modules.stt import transcribe_audio
from modules.rag import RagSystem, generate_response_stream
from modules.tts import synthesize_stream

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nct_robot_server")

app = FastAPI(title="NCT The Creepie Robot API")

# Khởi tạo RAG System (Load FAISS Database)
rag_system = RagSystem(index_path="faiss_index")

@app.get("/")
async def root():
    return {"message": "Welcome to NCT Robot Server"}

@app.websocket("/stream_audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via WebSocket.")
    try:
        while True:
            # 1. Thu nhận âm thanh (Từ Pi 4 gửi lên)
            data = await websocket.receive_bytes()
            logger.info(f"Received audio chunk of {len(data)} bytes")
            
            # 2. Xử lý STT Fast-Whisper
            text = transcribe_audio(data)
            
            if text:
                logger.info(f"Học sinh hỏi: {text}")
                
                # 3. RAG Retrieval (Truy xuất tài liệu từ FAISS)
                context = rag_system.retrieve(text)
                
                # 4. Sinh văn bản qua Llama.cpp (stream)
                text_stream = generate_response_stream(text, context)
                
                # Hàm helper chuyển text_stream sang async generator để dùng với asyncio pipeline
                async def async_text_stream(gen):
                    for chunk in gen:
                        if chunk:
                            yield chunk
                        await asyncio.sleep(0) # Tránh block event loop
                        
                async_gen = async_text_stream(text_stream)
                
                # 5. Tổng hợp giọng nói TTS Piper (stream) & Truyền thẳng về Pi
                async for audio_chunk in synthesize_stream(async_gen):
                    await websocket.send_bytes(audio_chunk)
                    
                logger.info("Finished responding to query.")
            
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
