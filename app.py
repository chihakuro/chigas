import gradio as gr
from transformers import pipeline

generator = pipeline('text-generation', model='deepseek-ai/DeepSeek-V2-Lite', trust_remote_code=True)

def generate_text(prompt, max_length):
    generated_text = generator(prompt, max_length=max_length, num_return_sequences=1)
    return generated_text[0]['generated_text']

iface = gr.Interface(
    fn=generate_text,
    inputs=[gr.Textbox(lines=2, placeholder="Nhập văn bản đầu vào"), gr.Slider(minimum=10, maximum=1000, value=50, label="Độ dài văn bản")],
    outputs="text",
    title="Ứng dụng Generative AI",
    description="Tạo văn bản với Llama3",
)

iface.launch(share=True) # share=True để triển khai lên Spaces
