import os
import torch
from torch.utils.data import Dataset, DataLoader

def map_labels_to_three_classes(label):
    if label in [0, 1]:
        return label
    else:
        return 2

class MultimodalDataset(Dataset):
    def __init__(self, start_dir, end_dir, image_dir):
        self.data = []

        for label_folder in os.listdir(start_dir):
            start_label_path = os.path.join(start_dir, label_folder)
            end_label_path = os.path.join(end_dir, label_folder)
            image_label_path = os.path.join(image_dir, label_folder)

            if not os.path.isdir(start_label_path):
                continue

            for fname in os.listdir(start_label_path):
                if not fname.endswith(".pt"):
                    continue

                start_path = os.path.join(start_label_path, fname)
                end_path = os.path.join(end_label_path, fname)
                image_path = os.path.join(image_label_path, fname)

                if os.path.exists(start_path) and os.path.exists(end_path) and os.path.exists(image_path):
                    label = int(label_folder.split("_")[0])  # "0_Normal" → 0
                    folder_path = os.path.join(label_folder, fname.replace(".pt", ""))  # replace ".pt" from folder name
                    self.data.append({
                        "start": start_path,
                        "end": end_path,
                        "image": image_path,
                        "label": label,
                        "folder":folder_path
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        start_data = torch.load(item["start"])  # dict
        end_data = torch.load(item["end"])      # dict
        image_tensor = torch.load(item["image"])  # tensor([512])

        start_text = start_data["text_emb"].float()  # [512]
        start_feat = start_data["feature"].squeeze(0).float()      # [9]
        end_emb = end_data["text_emb"].float()       # [512]
        image_emb = image_tensor.float()             # [512]

        # Map labels (Map 0→0, 1→1, Others→2)
        original_label = item["label"]
        mapped_label = map_labels_to_three_classes(original_label)
        
        label = torch.tensor(mapped_label, dtype=torch.long)

        return {
            "start": {
                "text_emb": start_text,
                "feature": start_feat
            },
            "end": end_emb,
            "image": image_emb,
            "label": label,
            "folder": item["folder"]  # Save folder name
        }
