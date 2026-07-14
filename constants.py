# constants.py

# --- Các icon và nhãn giao diện (UI) ---
EMOJI_CONF_QUANT = "🔄"
EMOJI_LOAD_TOKEN = "⏳"
EMOJI_LOAD_MODEL = "🤖"
EMOJI_SWAP = "🔄"

LABEL_SRC_EN = "Ngôn ngữ gốc: ANH"
LABEL_TGT_VI = "Ngôn ngữ đích: VIỆT"
LABEL_SRC_VI = "Ngôn ngữ gốc: VIỆT"
LABEL_TGT_EN = "Ngôn ngữ đích: ANH"

DIR_EN_TO_VI = "English ➔ Tiếng Việt"
DIR_VI_TO_EN = "Tiếng Việt ➔ English"

# --- Định dạng Prompt Template ---
PROMPT_EN_TO_VI = (
    "Role: You are a professional English-to-Vietnamese translator.\n"
    "Task: Translate the following source sentence accurately and naturally into Vietnamese.\n\n"
    "Source (English): {text}\n"
    "Target (Vietnamese):"
)

PROMPT_VI_TO_EN = (
    "Vai trò: Bạn là một chuyên gia dịch thuật Việt-Anh chuyên nghiệp.\n"
    "Nhiệm vụ: Hãy dịch câu gốc tiếng Việt sau đây sang tiếng Anh một cách chính xác và tự nhiên.\n\n"
    "Câu gốc (Tiếng Việt): {text}\n"
    "Câu dịch (Tiếng Anh):"
)