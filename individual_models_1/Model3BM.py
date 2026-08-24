import __main__
from pathlib import Path

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from drfp import DrfpEncoder


class MyNet(nn.Module):

    def __init__(self, input_shape):
        super(MyNet, self).__init__()

        self.fc_1 = nn.Linear(input_shape, 256)
        self.bn_1 = nn.BatchNorm1d(256)
        self.dropout_1 = nn.Dropout(p=0.25)

        self.fc_2 = nn.Linear(256, 128)
        self.bn_2 = nn.BatchNorm1d(128)
        self.dropout_2 = nn.Dropout(p=0.25)

        self.fc_3 = nn.Linear(128, 32)
        self.bn_3 = nn.BatchNorm1d(32)
        self.dropout_3 = nn.Dropout(p=0.25)

        self.fc_4 = nn.Linear(32, 1)

    def forward(self, input_data):

        x = F.relu(
            self.bn_1(
                self.fc_1(input_data)
            )
        )
        x = self.dropout_1(x)

        x = F.relu(
            self.bn_2(
                self.fc_2(x)
            )
        )
        x = self.dropout_2(x)

        x = F.relu(
            self.bn_3(
                self.fc_3(x)
            )
        )
        x = self.dropout_3(x)

        x = torch.sigmoid(
            self.fc_4(x)
        )

        x = x.squeeze(-1)

        return x




__main__.MyNet = MyNet


class Model3BM:

    def __init__(
        self,
        model_path="Model_3B-M.pth"
    ):

        model_dir = Path(__file__).resolve().parent
        model_path = model_dir / model_path

        self.device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        self.model = torch.load(
            str(model_path),
            map_location=self.device,
            weights_only=False
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def run(
        self,
        input_file,
        reaction_column,
        output_file,
        batch_size=256
    ):

        print(
            "========== Model_3B-M started =========="
        )

        # 只读取 Reaction 列
        df = pd.read_csv(
            input_file,
            usecols=[reaction_column]
        )

        reactions = df[reaction_column].tolist()

        print(
            f"Loaded {len(reactions)} reactions"
        )

        # =====================================================
        # DRFP
        # =====================================================

        print("Calculating DRFP...")

        fingerprints = DrfpEncoder.encode(
            reactions
        )

        X = np.asarray(
            fingerprints,
            dtype=np.float32
        )

        print(
            f"DRFP calculation completed: {X.shape}"
        )

        # =====================================================
        # DNN
        # =====================================================

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        preds = []
        probs = []

        with torch.no_grad():

            for i in range(
                0,
                len(X_tensor),
                batch_size
            ):

                batch = X_tensor[
                    i:i + batch_size
                ].to(self.device)

                prob = self.model(batch)

                prob = (
                    prob
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

                pred = (
                    prob >= 0.5
                ).astype(int)

                probs.extend(
                    prob.tolist()
                )

                preds.extend(
                    pred.tolist()
                )

        # =====================================================
        # 输出
        # =====================================================

        results = pd.DataFrame({
            "Reaction": reactions,
            "pred_label": preds,
            "pred_prob": probs
        })

        results.to_csv(
            output_file,
            index=False
        )

        print("Prediction completed.")
        print(results)

        print(
            f"Results have been saved to: "
            f"{output_file}"
        )

        return results