import logging
import subprocess
import asyncio

logger = logging.getLogger("tts_module")

# Giả định Piper binary đã có trong PATH và tải sẵn model tiếng việt ".onnx" tại dự án
PIPER_MODEL_PATH = "vi-model.onnx" 

async def synthesize_stream(text_stream):
    """
    Takes an async generator yielding text chunks and streams it to Piper TTS.
    Yields audio byte chunks back.
    Piper supports reading text from stdin and outputting raw PCM to stdout.
    """
    cmd = [
        "piper", 
        "--model", PIPER_MODEL_PATH, 
        "--output_raw"
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        
        async def feed_text():
            try:
                async for chunk in text_stream:
                    if chunk:
                        # Piper expects text followed by newline or sentences
                        process.stdin.write(chunk.encode("utf-8"))
                        await process.stdin.drain()
            finally:
                if process.returncode is None:
                    process.stdin.close()
            
        # Start feeding text in the background while we stream audio out
        feed_task = asyncio.create_task(feed_text())
        
        # Stream audio out
        while True:
            audio_chunk = await process.stdout.read(4096)
            if not audio_chunk:
                break
            # YIELD stream audio back to websocket
            yield audio_chunk
            
        await process.wait()
    except Exception as e:
        logger.error(f"TTS Streaming Error: {e}")
