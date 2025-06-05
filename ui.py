import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    StoppingCriteria
)
from peft import LoraConfig, PeftModel
import streamlit as st

# --- Configuration  ---
MAX_CTX_LEN = 1024
MAX_SUM_LEN = 25
MODEL_NAME = "distilgpt2"
DEFAULT_TIMESTAMP = "0602-0215" # Update this to your actual trained model's timestamp


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# --- Model Loading and Generation Functions ---
@st.cache_resource
def load_model_and_tokenizer(timestamp: str):

    st.write(f"Attempting to load model for timestamp: **{timestamp}**...")
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        st.warning("No GPU found. Model will load and run on CPU, which will be significantly slower.")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": device},
        quantization_config=bnb_config
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        f"models/lora-xsum-{timestamp}/adapters", # Path to your saved adapters
        device_map={"": device}
    )
    model.eval() 

    tok = AutoTokenizer.from_pretrained(f"models/lora-xsum-{timestamp}/tokenizer")
    tok.pad_token = tok.eos_token 

    st.write("Model and tokenizer loaded successfully on device:", device)
    return model, tok

def generate_summary(model, tokenizer, article: str, temperature: float = 0.8, top_p: float = 0.95) -> str:

    prompt = article + " TL;DR:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CTX_LEN
    ).to(model.device) 


    stop_words = ["\n", ".", "<EOS>"]
    stop_ids = []
    for word in stop_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[0])
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])


    out = model.generate(
        **inputs,
        max_new_tokens=MAX_SUM_LEN + 1, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        stopping_criteria=stopping_criteria
    )


    gen_ids = out[0, inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if "\n" in generated_text:
        generated_text = generated_text.split("\n")[0]

    return generated_text

# --- Streamlit UI ---
st.set_page_config(page_title="Summarization Chatbot", layout="wide")
st.title("💡 Summarization Chatbot")
st.markdown("---")

st.write(
    """
    This chatbot provides article summaries. 
    Simply enter your text, and the model will attempt to generate a summary.
    """
)

st.sidebar.header("Model Settings")
selected_timestamp = st.sidebar.text_input(
    "Model Timestamp (e.g., 0602-0215)",
    value=DEFAULT_TIMESTAMP,
    help="Corresponds to the timestamp part of your trained model's saved folder name."
)

with st.spinner(f"Loading model: {MODEL_NAME} ({selected_timestamp})..."):
    try:
        model, tokenizer = load_model_and_tokenizer(selected_timestamp)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}\nPlease check the path and timestamp. Ensure model files exist and you have sufficient GPU memory (or system RAM for CPU).")
        model = None
        tokenizer = None

st.sidebar.markdown("---")
st.sidebar.header("Generation Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.8, step=0.05,
                                help="Controls the randomness of the generated text. Higher values make the output more random.")
top_p = st.sidebar.slider("Top P", min_value=0.1, max_value=1.0, value=0.95, step=0.05,
                          help="Controls the diversity of the generated text. Only considers the highest probability tokens whose cumulative probability reaches Top P.")

st.markdown("---")

st.header("Enter Your Article")
user_input = st.text_area(
    "Paste or type the article content:",
    height=300,
    placeholder="Enter the article you want to summarize here..."
)

if st.button("Generate Summary"):
    if user_input and model and tokenizer:
        with st.spinner("Generating summary..."):
            generated_summary = generate_summary(model, tokenizer, user_input, temperature, top_p)
            st.success("Summary generation complete!")
            st.subheader("Generated Summary:")
            st.write(f"{generated_summary}")
    elif not user_input:
        st.warning("Please enter some text to generate a summary.")
    else:
        st.error("Model not loaded. Please resolve the model loading errors in the sidebar.")

st.markdown("---")

st.subheader("Try Some Example Articles:")
example_article_1 = """
Scientists recently announced a major breakthrough: they successfully corrected a gene defect associated with a rare genetic disease in laboratory conditions using a new gene-editing technology. 
This research opens new avenues for future disease treatments, but extensive clinical trials are still needed to ensure its safety and efficacy. This advancement marks a milestone in genetic medicine and promises to change the lives of millions of patients in the future.
"""
st.code(example_article_1, language="text")

example_article_2 = """
The global economy faces multiple challenges, including persistent inflationary pressures, geopolitical tensions, and supply chain disruptions. Governments and central banks worldwide are implementing measures to address these issues, such as raising interest rates to curb inflation and seeking new trade partners to enhance supply chain resilience. Despite uncertainties, some analysts believe the global economy still has potential for growth recovery with technological innovation and emerging markets expansion.
"""
st.code(example_article_2, language="text")