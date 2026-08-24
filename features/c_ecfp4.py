import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


class c_ECFP4:

    def c_ecfp4(self, reaction_smiles):

        try:
            reactants_str, product = reaction_smiles.split('>>')
            reactants = reactants_str.split('.')

        except:
            raise ValueError(
                f"Invalid reaction SMILES format: {reaction_smiles}"
            )

        product_mol = Chem.MolFromSmiles(product)

        if product_mol is None:
            raise ValueError(
                f"Unable to parse the SMILES of the product: {product}"
            )

        product_fp = AllChem.GetMorganFingerprintAsBitVect(
            product_mol,
            radius=2,
            nBits=1024,
            useChirality=True
        )

        combined_reactant_fp = np.zeros(
            1024,
            dtype=int
        )

        for smiles in reactants:

            reactant_mol = Chem.MolFromSmiles(smiles)

            if reactant_mol is None:
                raise ValueError(
                    f"Unable to parse the SMILES of the reactant(s): {smiles}"
                )

            reactant_fp = AllChem.GetMorganFingerprintAsBitVect(
                reactant_mol,
                radius=2,
                nBits=1024,
                useChirality=True
            )

            reactant_fp_array = np.array(
                list(reactant_fp.ToBitString()),
                dtype=int
            )

            combined_reactant_fp = np.bitwise_or(
                combined_reactant_fp,
                reactant_fp_array
            )

        combined_reactant_fp = combined_reactant_fp.astype(int)

        product_fp_array = np.array(
            list(product_fp.ToBitString()),
            dtype=int
        )

        reaction_fp = np.concatenate([
            combined_reactant_fp,
            product_fp_array
        ])

        return reaction_fp

    def process_reaction(self, reaction_smiles):

        try:

            fp = self.c_ecfp4(
                reaction_smiles
            )

            return pd.Series(fp)

        except Exception:

            return pd.Series(
                [np.nan] * 2048
            )

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

        print("========== c_ECFP4 started ==========")

        fp_df = df[reaction_column].apply(
            self.process_reaction
        )

        df = pd.concat(
            [
                df.reset_index(drop=True),
                fp_df.reset_index(drop=True)
            ],
            axis=1
        )

        if output_file is not None:

            df.to_csv(
                output_file,
                index=False
            )

            print(
                f"c_ECFP4 result saved to: {output_file}"
            )

        print("========== c_ECFP4 completed ==========")

        return df