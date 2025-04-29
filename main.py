# chat.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from config import MODEL_NAME, MAX_NEW_TOKENS, SYSTEM_PROMPT, SAVE_HISTORY, HISTORY_FILE
from save_chat import save_chat_history
from chat_memory import ChatMemory
import threading
import sys

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔧 디바이스 설정: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map=device
    )

    memory = ChatMemory(system_prompt=SYSTEM_PROMPT, tokenizer=tokenizer, max_total_tokens=2048)

    chat_log = []  # 파일로 저장할 때 쓸 기록
    print("🤖 EXAONE 메모리 챗봇에 오신 걸 환영합니다! (종료하려면 'exit' 입력)")

    try:
        while True:
            try:
                user_input = input("\n👤 당신: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Ctrl+C 감지. 챗봇을 종료합니다.")
                if SAVE_HISTORY and chat_log:
                    save_chat_history(chat_log, HISTORY_FILE)
                    print(f"💾 대화 기록이 {HISTORY_FILE} 파일에 저장되었습니다.")
                sys.exit(0)

            if not user_input:
                print("⚠️ 입력이 비어있습니다. 다시 입력해 주세요.")
                continue

            if user_input.lower() in ["exit", "quit", "종료"]:
                print("👋 챗봇을 종료합니다.")
                if SAVE_HISTORY and chat_log:
                    save_chat_history(chat_log, HISTORY_FILE)
                    print(f"💾 대화 기록이 {HISTORY_FILE} 파일에 저장되었습니다.")
                break

            # memory에 있는 대화 + 이번 질문을 합쳐서 모델 input 생성
            messages = memory.get_memory_with_prompt() + [{"role": "user", "content": user_input}]

            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)

                # 스트리머 생성
                streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

                # 모델 생성 호출을 별도의 스레드에서 실행
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

                # 스트리밍된 토큰 출력
                print("\n🤖 EXAONE: ", end="", flush=True)
                response = ""
                for token in streamer:
                    print(token, end="", flush=True)
                    response += token
                print()

                # 대화 기록 업데이트
                memory.add_turn(user_input, response)
                chat_log.append({"user": user_input, "exaone": response})

            except Exception as e:
                print(f"\n🚨 모델 생성 중 오류 발생: {e}")
                print("다시 시도해 주세요.")
                continue

    except Exception as e:
        print(f"\n🚨 예기치 못한 오류 발생: {e}")
        if SAVE_HISTORY and chat_log:
            save_chat_history(chat_log, HISTORY_FILE)
            print(f"💾 대화 기록이 {HISTORY_FILE} 파일에 저장되었습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
