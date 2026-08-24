import pandas as pd
import xgboost as xgb

from pathlib import Path

from features.c_ecfp4 import c_ECFP4





class Model1AM:

    def __init__(
        self,
        model_path="Model_1A-M.model"
    ):

        # Get model path relative to this Python file
        model_dir = Path(__file__).resolve().parent
        model_path = model_dir / model_path

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Initialize c_ECFP4
        self.c_ecfp4 = c_ECFP4()

    def run(
        self,
        input_file,
        reaction_column,
        output_file
    ):

        print("========== Model_1A-M started ==========")

        # Read only the reaction column
        df = pd.read_csv(
            input_file,
            usecols=[reaction_column]
        )

        reactions = df[reaction_column].tolist()

        print(f"Loaded {len(reactions)} reactions")

        # Generate c_ECFP4 fingerprints
        fingerprints = [
            self.c_ecfp4.c_ecfp4(reaction)
            for reaction in reactions
        ]

        X_valid = pd.DataFrame(fingerprints)


        # XGBoost prediction
        dtest = xgb.DMatrix(X_valid)

        probs = self.model.predict(dtest)

        preds = (probs > 0.5).astype(int)

        results = pd.DataFrame({
            'Standardized_reaction': reactions,
            'pred_label': preds,
            'pred_prob': probs
        })

        results.to_csv(
            output_file,
            index=False
        )

        print("Prediction completed.")
        print("Result:")
        print(results)
        print("Results have been saved.")

        return results