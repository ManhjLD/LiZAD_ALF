
import torch
from torch import nn
from torch.nn import functional as F
from .adapters import Adapter


class LayerWeightPredictor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        logits = self.network(features).squeeze(-1)
        return torch.softmax(logits, dim=1)


class ZSADModel(nn.Module):
    def __init__(self, args):
        super(ZSADModel, self).__init__()

        self.number_of_vision_layers = len(args.vision_layers)
        self.img_size = args.img_size

        self.fusion_beta = 0.6

        if not 0.0 <= self.fusion_beta <= 1.0:
            raise ValueError("fusion_beta must be between 0.0 and 1.0")

        text_dim = args.text_dim
        vision_dim = args.vision_dim
        out_dim = args.out_dim

        self.positive_text_adapter = Adapter(
            text_dim, out_dim, bottleneck=None, last_activation=True
        )
        self.negative_text_adapter = Adapter(
            text_dim, out_dim, bottleneck=None, last_activation=True
        )

        self.patch_adapter = nn.ModuleList([
            Adapter(
                vision_dim, out_dim,
                bottleneck=None,
                last_activation=True
            )
            for _ in range(self.number_of_vision_layers)
        ])

        self.cls_adapter = nn.ModuleList([
            Adapter(
                vision_dim, out_dim,
                bottleneck=None,
                last_activation=True
            )
            for _ in range(self.number_of_vision_layers)
        ])

        self.anomaly_layer_logits = nn.Parameter(
            torch.zeros(self.number_of_vision_layers)
        )

        self.global_layer_logits = nn.Parameter(
            torch.zeros(self.number_of_vision_layers)
        )

        self.anomaly_weight_predictor = LayerWeightPredictor(out_dim)
        self.global_weight_predictor = LayerWeightPredictor(out_dim)

    def text_guided_anomaly_scoring(
        self,
        patch_features,
        adapted_text_features
    ):
        batch_size, num_patches, _ = patch_features.shape
        grid_size = int(num_patches ** 0.5)

        scores = 100 * patch_features @ adapted_text_features

        scores = scores.permute(0, 2, 1).reshape(
            batch_size,
            2,
            grid_size,
            grid_size
        )

        scores = F.interpolate(
            scores,
            size=self.img_size,
            mode="bilinear",
            align_corners=True
        )

        return torch.softmax(scores, dim=1)

    def global_context_regularizer(
        self,
        cls_features,
        adapted_text_features
    ):
        return (
            100 * cls_features.unsqueeze(1) @ adapted_text_features
        ).squeeze(1)

    def static_dynamic_fusion(
        self,
        static_logits,
        dynamic_weights
    ):
        static_weights = torch.softmax(static_logits, dim=0)

        static_weights = static_weights.unsqueeze(0).expand_as(
            dynamic_weights
        )

        beta = self.fusion_beta

        fused_weights = (
            (1.0 - beta) * static_weights
            + beta * dynamic_weights
        )

        return fused_weights

    def forward(self, text_embeddings, image_features):
        adapted_positive_text_embeddings = self.positive_text_adapter(
            text_embeddings["normal"]
        )

        adapted_negative_text_embeddings = self.negative_text_adapter(
            text_embeddings["abnormal"]
        )

        adapted_text_features = torch.stack(
            [
                adapted_positive_text_embeddings,
                adapted_negative_text_embeddings
            ],
            dim=-1
        )

        cls, patches = image_features

        anomaly_maps = []
        global_scores = []
        adapted_patch_features = []
        adapted_cls_features = []

        for i in range(self.number_of_vision_layers):
            cls_features = self.cls_adapter[i](cls[:, i, :])
            cls_features = F.normalize(cls_features, dim=-1)

            patch_features = self.patch_adapter[i](
                patches[:, i, :, :]
            )
            patch_features = F.normalize(patch_features, dim=-1)

            adapted_cls_features.append(cls_features)
            adapted_patch_features.append(patch_features)

            anomaly_map = self.text_guided_anomaly_scoring(
                patch_features,
                adapted_text_features
            )

            global_score = self.global_context_regularizer(
                cls_features,
                adapted_text_features.detach()
            )

            anomaly_maps.append(anomaly_map)
            global_scores.append(global_score)

        anomaly_maps = torch.stack(anomaly_maps, dim=1)
        global_scores = torch.stack(global_scores, dim=1)

        patch_layer_features = torch.stack(
            adapted_patch_features,
            dim=1
        ).mean(dim=2)

        cls_layer_features = torch.stack(
            adapted_cls_features,
            dim=1
        )

        anomaly_dynamic_weights = self.anomaly_weight_predictor(
            patch_layer_features
        )

        global_dynamic_weights = self.global_weight_predictor(
            cls_layer_features
        )

        anomaly_weights = self.static_dynamic_fusion(
            self.anomaly_layer_logits,
            anomaly_dynamic_weights
        )

        global_weights = self.static_dynamic_fusion(
            self.global_layer_logits,
            global_dynamic_weights
        )

        anomaly_weights = anomaly_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        global_weights = global_weights.unsqueeze(-1)

        text_guided_anomaly_map = (
            anomaly_maps * anomaly_weights
        ).sum(dim=1)

        global_context_regularizer_score = (
            global_scores * global_weights
        ).sum(dim=1)

        return (
            text_guided_anomaly_map,
            global_context_regularizer_score
        )


