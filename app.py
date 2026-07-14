import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import toàn bộ icon, emoji và prompt từ file cấu hình của ông
import constants as c  

# 1. Cấu hình Quantization & Paths
BASE_MODEL_NAME = "./base_qwen"
ADAPTER_EN_VI = "./adapters/adapter_en_vi"
ADAPTER_VI_EN = "./adapters/adapter_vi_en"

print(f"{c.EMOJI_CONF_QUANT} Đang cấu hình Quantization 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. Load Tokenizer & Model
print(f"{c.EMOJI_LOAD_TOKEN} Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

print(f"{c.EMOJI_LOAD_MODEL} Loading Base Model 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_EN_VI, adapter_name="en_to_vi")
model.load_adapter(ADAPTER_VI_EN, adapter_name="vi_to_en")

# 3. Động cơ dịch thuật (sử dụng Prompt từ constants)
def translate_engine(text, direction):
    if not text.strip():
        return ""
        
    generation_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    if direction == c.DIR_EN_TO_VI:
        model.set_adapter("en_to_vi")
        prompt = c.PROMPT_EN_TO_VI.format(text=text)
    else:
        model.set_adapter("vi_to_en")
        prompt = c.PROMPT_VI_TO_EN.format(text=text)
        
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
        
    prompt_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][prompt_length:]
    
    translated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return translated_text.split('\n')[0].strip()

# 4. Logic đổi chiều trên Giao diện
def swap_languages(current_dir):
    if current_dir == c.DIR_EN_TO_VI:
        return c.DIR_VI_TO_EN, c.LABEL_SRC_VI, c.LABEL_TGT_EN, "", ""
    else:
        return c.DIR_EN_TO_VI, c.LABEL_SRC_EN, c.LABEL_TGT_VI, "", ""

# --- 5. CSS: giao diện dạng công cụ làm việc (tool), tối giản, không "màu mè AI" ---
custom_css = """
:root {
    --ink: #1c1c1e;
    --ink-soft: #57575c;
    --line: #dcdce0;
    --paper: #fafaf9;
    --panel: #ffffff;
    --accent: #2a3d66;
    --accent-hover: #1f2e4d;
}

body, .gradio-container {
    background-color: var(--paper) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    color: var(--ink) !important;
}

.gradio-container {
    max-width: 880px !important;
}

/* Thanh tiêu đề dạng toolbar, không phải "hero" căn giữa kiểu landing page */
.app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding: 0 0 14px 0;
    margin-bottom: 22px;
}
.app-header h1 {
    color: var(--ink) !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.01em;
    margin: 0 !important;
}
.app-header span.tag {
    color: var(--ink-soft) !important;
    font-size: 0.8rem !important;
    font-weight: 400;
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 2px 8px;
}

.translation-card {
    background: var(--panel) !important;
    border-radius: 6px !important;
    border: 1px solid var(--line) !important;
    padding: 0 !important;
    box-shadow: none !important;
    overflow: hidden;
}

/* Thanh chọn chiều dịch: giống thanh công cụ, không phải hai chữ "bay" giữa trang */
#lang-header-row {
    background-color: #f4f4f5 !important;
    border-bottom: 1px solid var(--line) !important;
    padding: 10px 16px !important;
    margin: 0 !important;
    gap: 8px !important;
}
.lang-label {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--ink-soft) !important;
    margin: 0 !important;
    display: flex;
    align-items: center;
    height: 100%;
}
.lang-label strong, .lang-label b { color: var(--ink) !important; }

.swap-btn {
    background-color: var(--panel) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
    max-width: 130px !important;
    margin: 0 auto !important;
}
.swap-btn:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: var(--panel) !important;
    transform: none !important;
}

.textbox-container {
    padding: 0 !important;
}
.textbox-container textarea {
    border-radius: 0 !important;
    border: none !important;
    border-right: 1px solid var(--line) !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    padding: 18px !important;
    background-color: var(--panel) !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
}
.textbox-container textarea:focus {
    border-color: var(--line) !important;
    background-color: #fbfbfa !important;
    box-shadow: none !important;
    outline: none !important;
}
.output-container textarea {
    background-color: #fafafa !important;
    color: var(--ink) !important;
    border-right: none !important;
}

.action-row {
    padding: 12px 16px !important;
    margin: 0 !important;
    border-top: 1px solid var(--line) !important;
    background-color: #fbfbfa !important;
}

.submit-btn {
    background: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border-radius: 4px !important;
    border: none !important;
    padding: 9px 22px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
}
.submit-btn:hover {
    background: var(--accent-hover) !important;
    transform: none !important;
    box-shadow: none !important;
    filter: none !important;
}
.submit-btn:active {
    transform: none !important;
    opacity: 0.9;
}
"""

# 6. Dựng giao diện Gradio với layout kiểu công cụ làm việc (tool-like), gọn và chuyên nghiệp
with gr.Blocks() as demo:

    gr.HTML(
        """
        <div class="app-header">
            <h1>AI Translator · Local</h1>
            <span class="tag">QLoRA · Multi-Adapter</span>
        </div>
        """
    )

    direction_state = gr.State(c.DIR_EN_TO_VI)

    with gr.Group(elem_classes="translation-card"):
        # Thanh chọn ngôn ngữ + nút đổi chiều
        with gr.Row(variant="compact", elem_id="lang-header-row"):
            src_label = gr.Markdown(f"{c.LABEL_SRC_EN}", elem_classes="lang-label")
            swap_btn = gr.Button(f"{c.EMOJI_SWAP} Đổi chiều", elem_classes="swap-btn")
            tgt_label = gr.Markdown(f"{c.LABEL_TGT_VI}", elem_classes="lang-label")

        # Hai ô nhập liệu song hành, sát cạnh nhau như một khối
        with gr.Row(equal_height=True):
            input_box = gr.Textbox(
                placeholder="Nhập văn bản cần dịch...",
                lines=8,
                label="",
                show_label=False,
                elem_classes="textbox-container"
            )
            output_box = gr.Textbox(
                placeholder="Bản dịch sẽ hiển thị ở đây...",
                lines=8,
                label="",
                show_label=False,
                interactive=False,
                elem_classes="textbox-container output-container"
            )

        # Thanh hành động dưới cùng, gọn, không nổi bật quá mức
        with gr.Row(elem_classes="action-row"):
            gr.Markdown("")
            translate_btn = gr.Button("Dịch", variant="primary", elem_classes="submit-btn")

    # Gán sự kiện cho các nút bấm
    translate_btn.click(
        fn=translate_engine,
        inputs=[input_box, direction_state],
        outputs=output_box
    )

    swap_btn.click(
        fn=swap_languages,
        inputs=[direction_state],
        outputs=[direction_state, src_label, tgt_label, input_box, output_box]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css
    )