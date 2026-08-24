
import pandas as pd

from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer
)


class RXNFP:

    def __init__(self):

        # Load default RXNFP model and tokenizer
        model, tokenizer = get_default_model_and_tokenizer()

        # Initialize RXNFP generator
        self.rxnfp_generator = RXNBERTFingerprintGenerator(
            model,
            tokenizer
        )

    def run(
        self,
        input_file,
        reaction_column="Standardized reaction",
        batch_size=100,
        output_file=None
    ):
        df = pd.read_csv(
            input_file,
            usecols=[reaction_column]
        )
        
        print("========== RXNFP started ==========")

        if reaction_column not in df.columns:

            raise ValueError(
                f"Column not found in DataFrame: {reaction_column}"
            )

        result_rows = []

        # Process reactions in batches
        for i in range(0, len(df), batch_size):

            batch_df = df.iloc[i:i + batch_size]

            reactions = batch_df[reaction_column].tolist()

            fingerprints = self.rxnfp_generator.convert_batch(
                reactions
            )

            # Generate fingerprint column names
            fp_columns = [
                f"rxnfp_{i + 1}"
                for i in range(len(fingerprints[0]))
            ]

            fp_df = pd.DataFrame(
                fingerprints,
                columns=fp_columns
            )

            # Combine original data and fingerprints
            batch_result = pd.concat(
                [
                    batch_df.reset_index(drop=True),
                    fp_df
                ],
                axis=1
            )

            result_rows.append(batch_result)

        result_df = pd.concat(
            result_rows,
            axis=0
        ).reset_index(drop=True)

        # Save output
        if output_file is not None:

            result_df.to_csv(
                output_file,
                index=False
            )

            print(
                f"RXNFP have been successfully generated "
                f"and saved to ... {output_file}"
            )

        print("========== RXNFP completed ==========")

        return result_df