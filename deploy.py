import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import os
MODEL_PATH = "./output/final_model"
MAX_NEW_TOKENS = 150
TOP_P = 0.92
TEMPERATURE = 0.7
HF_REPO_NAME = "dialogue-assistant-model"  # Change to your desired repository name

class ChatBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
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
    
    with gr.Blocks() as demo:
        gr.Markdown("# Dialogue Assistant")
        gr.Markdown("Chat with a fine-tuned language model specialized in dialogue responses")
        
        chatbot_interface = gr.ChatInterface(
            chatbot.generate_response,
            examples=[
                "Tell me about the solar system.",
                "What's the best way to learn a new language?",
                "Can you recommend a good science fiction book?"
            ],
            title="Dialogue Assistant",
        )
    
    return demo

def push_to_hub():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        return
    
    login(token=hf_token)
    api = HfApi()
    
    print(f"Pushing model to Hugging Face Hub as {HF_REPO_NAME}")
    api.create_repo(
        repo_id=HF_REPO_NAME,
        private=False,
        exist_ok=True
    )
    
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=HF_REPO_NAME,
        repo_type="model"
    )
    
    spaces_repo = f"{HF_REPO_NAME}-demo"
    api.create_repo(
        repo_id=spaces_repo,
        private=False,
        exist_ok=True,
        repo_type="space",
        space_sdk="gradio"
    )
    
    with open("requirements_spaces.txt", "w") as f:
        f.write("gradio>=3.50.2\n")
        f.write("torch>=2.0.0\n")
        f.write("transformers>=4.30.0\n")
        f.write(f"git+https://huggingface.co/{HF_REPO_NAME}\n")
    
    with open("app_spaces.py", "w") as f:
        f.write("""import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
MODEL_NAME = "{}"  # Your model name on Hugging Face
MAX_NEW_TOKENS = 150
TOP_P = 0.92
TEMPERATURE = 0.7

class ChatBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def format_prompt(self, message, history):
        prompt = ""
        for user_msg, bot_msg in history:
            prompt += f"Human: {{user_msg}}\\nAssistant: {{bot_msg}}\\n"
        prompt += f"Human: {{message}}\\nAssistant:"
        return prompt
    
    def generate_response(self, message, history):
        prompt = self.format_prompt(message, history)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = full_output.split("Assistant:")[-1].strip()
        
        # Handle potential incomplete responses
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        return response

# Create Gradio demo
chatbot = ChatBot()

with gr.Blocks() as demo:
    gr.Markdown("# Dialogue Assistant")
    gr.Markdown("Chat with a fine-tuned language model specialized in dialogue responses")
    
    chatbot_interface = gr.ChatInterface(
        chatbot.generate_response,
        examples=[
            "Tell me about the solar system.",
            "What's the best way to learn a new language?",
            "Can you recommend a good science fiction book?"
        ],
        title="Dialogue Assistant",
    )

demo.launch()""".format(HF_REPO_NAME))
    
    files_to_upload = ["app_spaces.py", "requirements_spaces.txt"]
    for file in files_to_upload:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file.replace("_spaces", "") if "_spaces" in file else file,
            repo_id=spaces_repo,
            repo_type="space"
        )
    
    print(f"Model pushed to: https://huggingface.co/{HF_REPO_NAME}")
    print(f"Demo app pushed to: https://huggingface.co/spaces/{spaces_repo}")

if __name__ == "__main__":
    # Deploy locally
    demo = create_demo()
    demo.launch()
    
    # push_to_hub()