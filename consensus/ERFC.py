from pathlib import Path
import pandas as pd


class ConsensusModel:

    def __init__(
        self,
        model1_result,
        model2_result,
        model3_result,
        model4_result
    ):

        self.model1_result = Path(model1_result)
        self.model2_result = Path(model2_result)
        self.model3_result = Path(model3_result)
        self.model4_result = Path(model4_result)

    def run(self, output_file):

        print("========== Consensus ERFC Model started ==========")

        # Check files
        for path in [
            self.model1_result,
            self.model2_result,
            self.model3_result,
            self.model4_result
        ]:

            if not path.exists():
                raise FileNotFoundError(
                    f"Prediction file not found:\n{path.resolve()}"
                )

        df1 = pd.read_csv(self.model1_result)
        df2 = pd.read_csv(self.model2_result)
        df3 = pd.read_csv(self.model3_result)
        df4 = pd.read_csv(self.model4_result)

        # =====================================================
        # Check required columns
        # =====================================================

        required_columns = [
            "Standardized_reaction",
            "pred_prob"
        ]

        for i, df in enumerate(
            [df1, df2, df3, df4],
            start=1
        ):

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Model {i} result is missing columns: "
                    f"{missing}"
                )

        # =====================================================
        # Check number of reactions
        # =====================================================

        if not (
            len(df1)
            == len(df2)
            == len(df3)
            == len(df4)
        ):

            raise ValueError(
                "The four model results contain "
                "different numbers of reactions."
            )

        # =====================================================
        # Check reaction order
        # =====================================================

        if not (
            df1["Standardized_reaction"].equals(df2["Standardized_reaction"])
            and
            df1["Standardized_reaction"].equals(df3["Standardized_reaction"])
            and
            df1["Standardized_reaction"].equals(df4["Standardized_reaction"])
        ):

            raise ValueError(
                "Reaction order/content is inconsistent "
                "between the four model results."
            )

        # =====================================================
        # Extract probabilities
        # =====================================================

        prob1 = df1["pred_prob"].to_numpy()
        prob2 = df2["pred_prob"].to_numpy()
        prob3 = df3["pred_prob"].to_numpy()
        prob4 = df4["pred_prob"].to_numpy()

        # =====================================================
        # Consensus probability
        # =====================================================

        consensus_prob = (
            prob1
            + prob2
            + prob3
            + prob4
        ) / 4.0

        # =====================================================
        # Consensus label
        # =====================================================

        consensus_label = (
            consensus_prob >= 0.5
        ).astype(int)

        # =====================================================
        # Build result
        # =====================================================

        results = pd.DataFrame({

            "Standardized_reaction": df1["Standardized_reaction"],

            "Model1AM_prob": prob1,

            "Model2AM_prob": prob2,

            "Model3BM_prob": prob3,

            "Model4BM_prob": prob4,

            "pred_prob": consensus_prob,

            "ERFC_pred_label": consensus_label

        })

        # =====================================================
        # Save
        # =====================================================

        results.to_csv(
            output_file,
            index=False
        )

        print("Consensus prediction completed.")

        print("Result:")
        print(results)

        print(
            f"Results have been saved to: "
            f"{output_file}"
        )

        print("========== Consensus ERFC Model completed ==========")

        return results