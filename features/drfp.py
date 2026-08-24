
import pandas as pd

from drfp import DrfpEncoder


class DRFP:

    def run(
        self,
        input_file,
        reaction_column="Standardized reaction",
        output_file=None
    ):
        
        df = pd.read_csv(
            input_file,
            usecols=[reaction_column]
        )

        print("========== DRFP started ==========")

        # Check whether the reaction column exists
        if reaction_column not in df.columns:

            raise ValueError(
                f"Column not found in DataFrame: {reaction_column}"
            )

        # Extract reaction SMILES
        rxn_list = df[reaction_column].tolist()

        # Calculate DRFP fingerprints
        fps = DrfpEncoder.encode(rxn_list)

        # Convert fingerprints to DataFrame
        if len(fps) > 0:

            n_bits = len(fps[0])

            fp_df = pd.DataFrame(
                fps,
                columns=range(n_bits)
            )

        else:

            fp_df = pd.DataFrame()

        # Reset index before concatenation
        df_reset = df.reset_index(drop=True)

        fp_df_reset = fp_df.reset_index(drop=True)

        # Combine original data and fingerprints
        result_df = pd.concat(
            [
                df_reset,
                fp_df_reset
            ],
            axis=1
        )

        # Save output
        if output_file is not None:

            result_df.to_csv(
                output_file,
                index=False
            )

            print(
                f"DRFP have been successfully generated "
                f"and saved to ... {output_file}"
            )

        print("========== DRFP completed ==========")

        return result_df
