import os
import re
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from bs4 import BeautifulSoup

IMPORTANT_TAGS = ['title', 'h1', 'h2', 'h3', 'p', 'a', 'span', 'button', 'li']

# Set Path
ROOT_DIR = Path(__file__).resolve().parent.parent

base_dir = ROOT_DIR / "data"
output_dir = ROOT_DIR / "pt_data" / "koclip_pt_end"
parser_error_log_path = ROOT_DIR / "parser_error_log_end.txt"

# ✅ Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load KoCLIP
model_name = "koclip/koclip-base-pt"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


# Refine Text
def clean_text(text):
    text = re.sub(r'[^\w\s가-힣]', ' ', text)   # Remove special characters (allow only Korean, numbers, and English letters)
    text = re.sub(r'\s+', ' ', text).strip()   # Strip whitespace
    return text

# Parse HTML from text from end.txt
def extract_text(html: str, max_char_len: int = 2048) -> str:
    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        soup = BeautifulSoup(html, 'html.parser')

    texts = []
    for tag in IMPORTANT_TAGS:
        for elem in soup.find_all(tag):
            txt = elem.get_text(strip=True)
            if txt and len(txt) > 1:
                texts.append(txt)
            if sum(len(t) for t in texts) > max_char_len:
                break
        if sum(len(t) for t in texts) > max_char_len:
            break

    combined = " ".join(texts)
    cleaned = clean_text(combined)

    return cleaned


def preprocess_text_only_end(data_root="./data_all", save_root="./pt_data_all/koclip_text_only_end"):
    all_paths = []
    for label_folder in os.listdir(data_root):
        if "_" not in label_folder:
            continue
        label_idx = label_folder.split("_")[0]
        if not label_idx.isdigit():
            continue

        label_path = os.path.join(data_root, label_folder)
        for site in os.listdir(label_path):
            site_path = os.path.join(label_path, site)
            end_path = os.path.join(site_path, "end.txt")
            save_path = os.path.join(save_root, label_folder, f"{site}.pt")

            if not os.path.exists(end_path):
                continue
            all_paths.append((label_idx, end_path, save_path))

    os.makedirs(save_root, exist_ok=True)
    open(parser_error_log_path, "w").close()

    for label_idx, end_path, save_path in tqdm(all_paths, desc="📄 Embedding end.txt ..."):
        try:
            with open(end_path, 'r', encoding='utf-8') as f:
                html = f.read()
        except Exception as e:
            print(f"❌ Error! Cannot open file: {end_path} → {e}")
            continue

        try:
            text = extract_text(html)
        except Exception as e:
            print(f"❌ Error! Cannot parse: {end_path} → {e}")
            with open(parser_error_log_path, "a", encoding="utf-8") as logf:
                logf.write(f"{end_path}\n")
            continue

        try:
            if len(text) > 512:
                text = text[:512]  # ⚠️ Cut off too long text

            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ send to GPU
            
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                text_emb = outputs.squeeze(0).cpu() # save in CPU

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"text_emb": text_emb, "label": int(label_idx)}, save_path)
            # print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Failed to Embed: {end_path} → {e}")
            continue



# ✅ Execute
preprocess_text_only_end(data_root=base_dir, save_root=output_dir)
