import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForTokenClassification, AutoTokenizer, pipeline
import gradio as gr

# Load GPT-2 model and tokenizer
MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Load BioBERT for medical NLP without requiring pip install
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
biobert_model = AutoModelForTokenClassification.from_pretrained(BIOBERT_MODEL)
nlp_pipeline = pipeline("ner", model=biobert_model, tokenizer=biobert_tokenizer)

# Conversation memory
conversation_history = []

# Function to generate text with medical NLP support
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9, clean_output=False, stopwords="", num_responses=1, extract_entities=False):
    global conversation_history
    
    # Combine conversation history with new prompt
    full_prompt = "\n".join(conversation_history + [prompt])
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    responses = []
    for _ in range(num_responses):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Text cleaning
        if clean_output:
            text = text.replace("\n", " ").strip()
        
        # Stopword filtering
        for word in stopwords.split(","):
            text = text.replace(word.strip(), "")
        
        # Extract medical entities using BioBERT
        if extract_entities:
            entities = nlp_pipeline(text)
            extracted_entities = set([entity["word"] for entity in entities])
            text += "\n\nExtracted Medical Entities:\n" + "\n".join(extracted_entities)
        
        responses.append(text)
    
    # Update conversation history
    conversation_history.append(prompt)
    conversation_history.append(responses[0])  # Store only the first response
    
    return "\n\n".join(responses)

# Gradio Interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Medical Query"),
        gr.Slider(50, 500, step=10, label="Max Length"),
        gr.Slider(0.1, 1.5, step=0.1, label="Temperature"),
        gr.Slider(0, 100, step=5, label="Top-K"),
        gr.Slider(0.0, 1.0, step=0.1, label="Top-P"),
        gr.Checkbox(label="Clean Output"),
        gr.Textbox(label="Stopwords (comma-separated)"),
        gr.Slider(1, 5, step=1, label="Number of Responses"),
        gr.Checkbox(label="Extract Medical Entities")
    ],
    outputs=gr.Textbox(label="Generated Medical Text"),
    title="Medical AI Assistant",
    description="Enter a medical-related prompt and adjust parameters to generate AI-assisted text. Supports entity recognition for medical terms.",
)

# Launch the app
demo.launch()
