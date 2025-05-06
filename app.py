import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./output/final_model"
MAX_NEW_TOKENS = 150
TOP_P = 0.92
TEMPERATURE = 0.7

class ChatBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.chat_history = []
    
    def format_prompt(self, message, history):
        prompt = ""
        for user_msg, bot_msg in history:
            prompt += f"Human: {user_msg}\nAssistant: {bot_msg}\n"
        prompt += f"Human: {message}\nAssistant:"
        return prompt
    
    def generate_response(self, message, history):
        prompt = self.format_prompt(message, history)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = full_output.split("Assistant:")[-1].strip()
        
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        return response

def create_demo():
    chatbot = ChatBot()
    
    demo = gr.Interface(
        fn=chatbot.generate_response,
        inputs=[
            gr.Textbox(lines=2, placeholder="Type your message here..."),
            gr.State([])
        ],
        outputs=[
            gr.Textbox(label="Response"),
            gr.State()
        ],
        title="Dialogue Assistant",
        description="A fine-tuned language model for dialogue responses",
        allow_flagging="never",
        examples=[
            ["Tell me about the solar system."],
            ["What's the best way to learn a new language?"],
            ["Can you recommend a good science fiction book?"]
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()