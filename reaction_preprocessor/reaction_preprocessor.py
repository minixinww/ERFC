
import pandas as pd

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import rdBase

from rxnmapper import RXNMapper

import warnings
from transformers import logging as hf_logging


class ReactionPreprocessor:

    def __init__(self):

        # Disable RDKit info and debug logs
        rdBase.DisableLog('rdApp.info')
        rdBase.DisableLog('rdApp.debug')

        # Disable Transformers warnings
        hf_logging.set_verbosity_error()

        warnings.filterwarnings(
            "ignore",
            message=".*AlbertSdpaAttention is used but.*"
        )

        # Initialize RXNMapper
        self.rxn_mapper = RXNMapper()

    # ============================================================
    # 1. Neutralize atoms
    # ============================================================

    def neutralize_atoms(self, mol):

        pattern = Chem.MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
        )

        at_matches = mol.GetSubstructMatches(pattern)

        at_matches_list = [y[0] for y in at_matches]

        if len(at_matches_list) > 0:

            for at_idx in at_matches_list:

                atom = mol.GetAtomWithIdx(at_idx)

                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()

                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()

        return mol

    # ============================================================
    # 2. Process single SMILES
    # ============================================================

    def process_smiles(self, smiles):

        if not isinstance(smiles, str) or smiles.strip() == '':
            return smiles

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:

            mol = rdMolStandardize.Cleanup(mol)

            mol = self.neutralize_atoms(mol)

            canonical_smiles = Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                canonical=True
            )

            return canonical_smiles

        else:

            print(f"Unable to parse SMILES: {smiles}")

            return smiles

    # ============================================================
    # 3. Standardize reaction SMILES
    # ============================================================

    def process_reaction_smiles(self, reaction_smiles):

        if not isinstance(reaction_smiles, str):
            return reaction_smiles

        try:

            reactants_str, products_str = reaction_smiles.split('>>')

        except ValueError:

            print(
                f"Unable to parse reaction SMILES: {reaction_smiles}"
            )

            return reaction_smiles

        reactants = [
            self.process_smiles(x.strip())
            for x in reactants_str.split('.')
            if x.strip()
        ]

        products = [
            self.process_smiles(x.strip())
            for x in products_str.split('.')
            if x.strip()
        ]

        # Remove [H+]
        reactants = [
            mol for mol in reactants
            if mol != '[H+]'
        ]

        products = [
            mol for mol in products
            if mol != '[H+]'
        ]

        reactants.sort()
        products.sort()

        return '>>'.join([
            '.'.join(reactants),
            '.'.join(products)
        ])

    # ============================================================
    # 4. RXNMapper
    # ============================================================

    def process_reaction(self, reaction_smiles):

        try:

            reactants_list, products_list = reaction_smiles.split('>>')

            reactants = reactants_list.split(".")

            reactants.sort()

            products = products_list.split(".")

            # Generate atom-mapped reaction
            mapped_reactions = (
                self.rxn_mapper
                .get_attention_guided_atom_maps(
                    [reaction_smiles]
                )
            )

            mapped_reaction = mapped_reactions[0]['mapped_rxn']

            mapped_reactants_list, mapped_products_list = (
                mapped_reaction.split('>>')
            )

            mapped_reactants = mapped_reactants_list.split(".")

            mapped_products = mapped_products_list.split(".")

            # Check whether products were correctly mapped
            if products == mapped_products:

                print("Warning: Unmapped products detected")

                return ''

            else:

                correctly_mapped_reactants = []

                unmapped_reactants = []

                atom_mapped_reactants = []

                for i in range(len(reactants)):

                    if reactants[i] != mapped_reactants[i]:

                        correctly_mapped_reactants.append(
                            reactants[i]
                        )

                        atom_mapped_reactants.append(
                            mapped_reactants[i]
                        )

                    else:

                        unmapped_reactants.append(
                            reactants[i]
                        )

                correctly_mapped_reactants.sort()

                new_reaction = (
                    '.'.join(correctly_mapped_reactants)
                    + '>>'
                    + products_list
                )

                return new_reaction

        except Exception:

            return None

    # ============================================================
    # 5. Complete preprocessing
    # ============================================================

    def run(
        self,
        input_file,
        reaction_column,
        output_file=None
    ):

        print("========== Preprocessing started ==========")

        # Read input CSV
        df = pd.read_csv(input_file)

        print(f"Loaded {len(df)} rows")

        # ========================================================
        # Step 1: Standardize reaction SMILES
        # ========================================================

        print("Starting SMILES standardization...")

        df[reaction_column] = df[reaction_column].apply(
            self.process_reaction_smiles
        )

        print("SMILES standardization completed")

        # ========================================================
        # Step 2: RXNMapper
        # ========================================================

        print("Starting RXNMapper...")

        df['Standardized reaction'] = df[reaction_column].apply(
            self.process_reaction
        )

        print("RXNMapper completed")

        if output_file is not None:

            df.to_csv(
                output_file,
                index=False
            )

        print(
            f"Standardized data saved to: {output_file}"
        )


        print("========== Preprocessing completed ==========")

        # Return DataFrame for subsequent processing
        return df

