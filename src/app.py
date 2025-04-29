# main.py
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi.responses import StreamingResponse
import threading
from chat_memory import ChatMemory
from config import MODEL_NAME, SYSTEM_PROMPT, MAX_NEW_TOKENS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

# 서버 시작할 때 모델, 토크나이저 미리 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    device_map=device
)

memory = ChatMemory(system_prompt=SYSTEM_PROMPT, tokenizer=tokenizer, max_total_tokens=2048)

# 요청 body 스키마
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        user_input = req.user_input.strip()

        if not user_input:
            raise HTTPException(status_code=400, detail="입력이 비어 있습니다.")

        messages = memory.get_memory_with_prompt() + [{"role": "user", "content": user_input}]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # 답변 생성
        outputs = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        memory.add_turn(user_input, decoded)

        return {"response": decoded}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 새로운 /chat/stream 엔드포인트 (Streaming)
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    try:
        user_input = req.user_input.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="입력이 비어 있습니다.")

        messages = memory.get_memory_with_prompt() + [{"role": "user", "content": user_input}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        generation_thread = threading.Thread(
            target=model.generate,
            kwargs={
                "input_ids": input_ids,
                "eos_token_id": tokenizer.eos_token_id,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "streamer": streamer,
            },
        )
        generation_thread.start()

        def token_streamer():
            response_text = ""
            for token in streamer:
                yield token
                response_text += token
            memory.add_turn(user_input, response_text)

        return StreamingResponse(token_streamer(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

