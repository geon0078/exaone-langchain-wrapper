from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatEXAONE(BaseChatModel):
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

    def _generate(self, messages, stop=None):
        chat_template = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_template.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat_template.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_template.append({"role": "assistant", "content": msg.content})

        input_ids = self.tokenizer.apply_chat_template(
            chat_template,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            input_ids.to("cuda"),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return AIMessage(content=decoded)
