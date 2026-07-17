import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import toàn bộ icon, emoji và prompt từ file cấu hình
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
    # lượng tử hóa
    # Lần 1: Nén trọng số mô hình từ 16-bit xuống 4-bit
    # Lần 2: Nén các hằng số của danh sách quản lý việc nén lần 1 từ 32-bit xuống 8-bit
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

# 5. Dựng giao diện Gradio với layout kiểu công cụ làm việc (tool-like)
# Đường dẫn file style.css được truyền trực tiếp vào tham số `css`
with gr.Blocks(css="style.css") as demo:

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
        server_port=7860
    )