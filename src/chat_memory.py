# chat_memory.py

class ChatMemory:
    def __init__(self, system_prompt, tokenizer, max_total_tokens=2048):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_total_tokens = max_total_tokens
        self.history = []

    def add_turn(self, user_input, ai_output):
        """대화에 새로운 발화(user + AI)를 추가한다."""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": ai_output})

    def get_memory_with_prompt(self):
        """시스템 프롬프트 + 대화 히스토리 반환 (토큰 초과 관리 포함)"""
        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        # 토크나이즈해서 전체 토큰 길이 확인
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        total_tokens = input_ids.shape[-1]

        # 토큰 수가 초과되면 오래된 메시지부터 삭제
        while total_tokens > self.max_total_tokens and len(self.history) > 2:
            # user + assistant turn 하나 삭제
            self.history = self.history[2:]
            messages = [{"role": "system", "content": self.system_prompt}] + self.history
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            total_tokens = input_ids.shape[-1]

        return messages
