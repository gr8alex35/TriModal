import torch
import torch.nn as nn
import torch.nn.functional as F


# Start + Feature Fusion
class TextFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, text_emb, feature):
        f_proj = self.feature_proj(feature)
        x = torch.cat([text_emb, f_proj], dim=1)
        return self.fusion(x)

# Gated Attention Block (image → target)
class GatedAttentionBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, target):  # query: image, target: start or end
        Q = self.query_proj(query)    # [B, 512]
        K = self.key_proj(target)
        V = self.value_proj(target)

        attn_score = F.cosine_similarity(Q, K, dim=-1).unsqueeze(-1)  # [B, 1]
        attn_output = attn_score * V  # soft-attend value

        gate = self.gate(torch.cat([query, attn_output], dim=1))
        gated = gate * attn_output + (1 - gate) * query

        return self.out_proj(gated)

# TriModal
class TriModalClassifier(nn.Module):
    def __init__(self, use_start=True, use_end=True, use_image=True, num_classes=3):
        super().__init__()
        self.use_start = use_start
        self.use_end = use_end
        self.use_image = use_image

        if use_start:
            self.text_start_fusion = TextFeatureFusion()

        if use_end:
            self.end_proj = nn.Linear(512, 512)

        if use_image:
            self.image_proj = nn.Linear(512, 512)

        # GatedAttention: End → Start, End → Image
        self.attn_s2e = GatedAttentionBlock(512)  # End -> Start
        self.attn_i2e = GatedAttentionBlock(512)  # End -> Image

        in_dim = 0
        if use_image:
            in_dim += 512
        if use_start:
            in_dim += 512
        if use_end:
            in_dim += 512

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, start_text=None, start_feat=None, end_text=None, image_emb=None):
        features = []
        image_proj = None

        # Step 1: Project end text
        if self.use_end:
            end_proj = self.end_proj(end_text)
            features.append(end_proj)

        # Step 2: Start -> End Attention
        if self.use_start:
            start_fused = self.text_start_fusion(start_text, start_feat)  # shape: (B, 512)
            if self.use_end:                
                attn_s2e = self.attn_s2e(end_proj, start_fused)
                features.append(attn_s2e)
            else:
                features.append(start_fused)

        # Step 3: Image -> End Attention
        if self.use_image:
            image_proj = self.image_proj(image_emb)
            if self.use_end:
                attn_i2e = self.attn_i2e(end_proj, image_proj)
                features.append(attn_i2e)
            else:
                features.append(image_proj)

        # Step 4: Feature concat → classification
        x = torch.cat(features, dim=1)
        return self.classifier(x)