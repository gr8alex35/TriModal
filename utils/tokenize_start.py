import os
import re
import torch
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoProcessor, AutoModel

IMPORTANT_TAGS = ['title', 'h1', 'h2', 'h3', 'p', 'a', 'span', 'button', 'li']
FEATURE_ORDER = [...]

# Set Path
ROOT_DIR = Path(__file__).resolve().parent.parent

base_dir = ROOT_DIR / "data"
output_dir = ROOT_DIR / "pt_data" / "koclip_pt_start"
parser_error_log_path = ROOT_DIR / "parser_error_log_start.txt"

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

# Parse HTML from text from start.txt
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

def extract_feature(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    script_tags = soup.find_all('script')
    meta_tags = soup.find_all('meta')
    script_texts = [tag.get_text() for tag in script_tags if tag.get_text()]
    full_script = "\n".join(script_texts)
    return torch.tensor([
        int("window.location" in full_script),
        int("document.location" in full_script),
        int("top.location" in full_script),
        int("window.open" in full_script),
        int("eval(" in full_script),
        int(any("http-equiv" in tag.attrs and "refresh" in str(tag.attrs.get("http-equiv", "")).lower() for tag in meta_tags)),
        len(script_tags),
        max([len(s) for s in script_texts], default=0) / 1000.0,
        (
            full_script.count(')') + full_script.count('}') + full_script.count('=') +
            len(re.findall(r'atob\s*\(', full_script)) +
            len(re.findall(r'btoa\s*\(', full_script)) +
            len(re.findall(r'unescape\s*\(', full_script)) +
            len(re.findall(r'eval\(function\(p,a,c,k,e,r\)', full_script)) +
            len([w for w in full_script.split() if len(w) >= 512])
        ) / 100.0
    ], dtype=torch.float)

def preprocess_all(data_root="./data_all", save_root="./pt_data_all/koclip_pt_start"):
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
            start_path = os.path.join(site_path, "start.txt")
            save_path = os.path.join(save_root, label_folder, f"{site}.pt")

            if not os.path.exists(start_path):
                continue
            all_paths.append((label_folder, label_idx, site, start_path, save_path))

    os.makedirs(save_root, exist_ok=True)
    open(parser_error_log_path, "w").close()

    for label_folder, label_idx, site, start_path, save_path in tqdm(all_paths, desc="📄 Processing start.txt"):
        try:
            with open(start_path, 'r', encoding='utf-8') as f:
                html = f.read()
        except Exception as e:
            print(f"❌ Error! Cannot open file: {start_path} → {e}")
            continue

        try:
            text = extract_text(html)
        except Exception as e:
            print(f"❌ Error! Failed to parse: {start_path} → {e}")
            with open(parser_error_log_path, "a", encoding="utf-8") as logf:
                logf.write(f"{start_path}\n")
            continue

        try:
            if len(text) > 512:
                text = text[:512]  # ⚠️ Cut off too long text

            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ send to GPU

            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                text_emb = outputs.squeeze(0).cpu()  # save in CPU [512]

            # Extract features from HTML
            feature = extract_feature(html).unsqueeze(0)  # [9] → [1, 9]

            # Save
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "text_emb": text_emb,        # [512]
                "feature": feature,          # [1, 9]
                "label": int(label_idx)
            }, save_path)

            # print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Failed to Embed: {start_path} → {e}")
            continue






# ✅ Execute
preprocess_all(data_root=base_dir, save_root=output_dir)