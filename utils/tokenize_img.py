import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# ✅ Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model_name = "koclip/koclip-base-pt"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Set Path
ROOT_DIR = Path(__file__).resolve().parent.parent

base_dir = ROOT_DIR / 'data'
output_root = ROOT_DIR / 'pt_data' / 'koclip_pt_image'

# image embedder
def extract_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # send to GPU

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)  # shape: (1, 512)

    embedding = outputs.squeeze(0).cpu()  # save in CPU
    if embedding.shape != torch.Size([512]):
        raise ValueError(f"❌ Error! Embedding size: {image_path} → {embedding.shape}")
    return embedding




# ✅ Execute
folders = [f for f in base_dir.glob("*_*/*") if f.is_dir()]
print(f"📁 Total number of images: {len(folders)}")

for folder in tqdm(folders, desc="🖼️ Embedding images ..."):
    name = folder.name
    label_folder = folder.parent.name
    image_path = folder / "end_screenshot.png"

    if not image_path.exists():
        print(f"⚠️ {name}: end_screenshot.png doesn't exist → Skip")
        continue

    try:
        embedding = extract_image_embedding(image_path)
    except Exception as e:
        print(f"❌ Error!: {name} - {e}")
        continue

    # 저장 경로 설정
    save_dir = output_root / label_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}.pt"
    torch.save(embedding, save_path)
    # print(f"✅ Saved: {save_path}")