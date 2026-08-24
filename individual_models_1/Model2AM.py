import pandas as pd
import xgboost as xgb

from pathlib import Path

from features.rxnfp import RXNFP


class Model2AM:

    def __init__(
        self,
        model_path="Model_2A-M.model"
    ):

        # Get model path relative to this Python file
        model_dir = Path(__file__).resolve().parent
        model_path = model_dir / model_path

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Initialize RXNFP
        self.rxnfp = RXNFP()

    def run(
        self,
        input_file,
        reaction_column,
        output_file
    ):

        print("========== Model_2A-M started ==========")

        # Generate RXNFP
        fp_df = self.rxnfp.run(
            input_file=input_file,
            reaction_column=reaction_column
        )

        # First column is reaction,
        # remaining columns are fingerprints
        X_valid = fp_df.iloc[:, 1:]

        reactions = fp_df.iloc[:, 0].tolist()



        # XGBoost prediction
        dtest = xgb.DMatrix(X_valid)

        probs = self.model.predict(dtest)

        preds = (probs > 0.5).astype(int)

        results = pd.DataFrame({
            'Reaction': reactions,
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