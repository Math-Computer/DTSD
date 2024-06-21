import timm
import torch
import torch.nn.functional as F
from typing import List
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm


class MemoryBank:
    def __init__(self, 
                 backbone: str, 
                 layers: List[int]) -> None:
        
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=layers).cuda()
        self.layers = layers
        self.memory_banks={}
        self.banksize: int

    def make(self, dataloader: DataLoader) -> None:
        self.feature_extractor.eval()
        
        with torch.no_grad():
            for data in tqdm(dataloader, desc="making bank"):
                input = data["img_origin"].cuda()
                features = self.feature_extractor(input)

                for i, layer in enumerate(self.layers):
                    if f"layer{layer}" not in self.memory_banks.keys():
                        self.memory_banks[f"layer{layer}"] = features[i]
                    else:
                        self.memory_banks[f"layer{layer}"] = torch.cat([self.memory_banks[f"layer{layer}"], features[i]], dim=0)      
        self.banksize = len(self.memory_banks["layer1"])

    def select(self, features: Tensor, layer: str) -> None:
        diff_matrix = self._calculate_diff(features, layer)

        selected_features = torch.index_select(
            input=self.memory_banks[layer],
            dim=0,
            index=diff_matrix.argmin(dim=1)
        )
        return selected_features

    def _calculate_diff(self, batch_features: Tensor, layer: str) -> Tensor:
        # 计算一个batch图像某个layer的特征与对应bank中的距离
        diff_matrix = torch.zeros(batch_features.size(0), self.banksize).cuda()

        for i, feature in enumerate(batch_features):
            diff = F.mse_loss(
                input=torch.repeat_interleave(feature.unsqueeze(0), repeats=self.banksize, dim=0),
                target=self.memory_banks[layer],
                reduction='none'
            ).mean(dim=[1, 2, 3])

            diff_matrix[i] = diff
        
        return diff_matrix


# dataset = MVTecDataset(
#     is_train=True,
#     mvtec_dir="datasets/mvtec/bottle/train/good/",
#     dtd_dir="datasets/dtd/images/",
# )
# dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

# mem = MemoryBank("resnet18", [1, 2, 3, 4])
# mem.make(dataloader)

# feature_extractor = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=[1, 2, 3, 4]).cuda()
# feature_extractor.eval()
# for data in dataloader:
#     input = data["img_origin"].cuda()
#     x1, x2, x3, x4 = feature_extractor(input)
#     print("x1", x1.size())
#     print("x2", x2.size())
#     print("x3", x3.size())
#     print("x4", x4.size())

#     x = mem.select(x1, "layer1")

#     break
