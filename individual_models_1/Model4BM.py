import pandas as pd
import torch

from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


class Model4BM:

    def __init__(
        self,
        model_path="Model_4B-M"
    ):

        # Model path relative to this Python file
        model_dir = Path(__file__).resolve().parent
        model_path = model_dir / model_path

        # Device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path)
        )

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model_4B-M loaded successfully.")

    def run(
        self,
        input_file,
        reaction_column="Standardized reaction",
        output_file=None,
        batch_size=32
    ):

        print(
            "========== Model_4B-M started =========="
        )

        # -----------------------------------------
        # Only read the reaction column
        # -----------------------------------------

        df = pd.read_csv(
            input_file,
            usecols=[reaction_column]
        )

        reactions = df[reaction_column].fillna("").tolist()

        print(
            f"Loaded {len(reactions)} reactions"
        )

        # -----------------------------------------
        # Prediction
        # -----------------------------------------

        all_probs = []
        all_preds = []

        for i in range(
            0,
            len(reactions),
            batch_size
        ):

            batch_reactions = reactions[
                i:i + batch_size
            ]

            inputs = self.tokenizer(
                batch_reactions,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

                logits = outputs.logits

                # Binary classification
                if logits.shape[-1] == 1:

                    probs = torch.sigmoid(
                        logits
                    ).squeeze(-1)

                else:

                    probs = torch.softmax(
                        logits,
                        dim=-1
                    )[:, 1]

                preds = (
                    probs >= 0.5
                ).long()

            all_probs.extend(
                probs.cpu().numpy().tolist()
            )

            all_preds.extend(
                preds.cpu().numpy().tolist()
            )

            print(
                f"Processed "
                f"{min(i + batch_size, len(reactions))}"
                f"/{len(reactions)}"
            )

        # -----------------------------------------
        # Results
        # -----------------------------------------

        results = pd.DataFrame({

            "Reaction": reactions,

            "pred_label": all_preds,

            "pred_prob": all_probs

        })

        # -----------------------------------------
        # Save
        # -----------------------------------------

        if output_file is not None:

            results.to_csv(
                output_file,
                index=False
            )

            print(
                f"Results have been saved to: "
                f"{output_file}"
            )

        print("Prediction completed.")
        print("Result:")
        print(results)

        print(
            "========== Model_4B-M completed =========="
        )

        return results