# ERFC

## Enzymatic Reaction Feasibility Classification

We developed a consensus model named Enzymatic Reaction Feasibility Classification (Consensus_ERFC) by averaging the predictions from four individual models: Model 1A-M (c_ECFP4-XGBoost), 2A-M (RXNFP-XGBoost), 3B-M (DRFP-DNN), and 4B-M (SMILES-based ChemMLM). This model is capable of handling both stereochemistry-preserving and stereochemistry-agnostic reactions, enabling the efficient and rapid screening of large-scale biosynthetic reactions

## Workflow

The complete prediction workflow consists of the following steps:

1. **Reaction preprocessing**
   - Standardize reactions and retain substrates containing products and product atoms via atom mapping.

2. **Individual model prediction**
   - Model 1A-M generates predictions using c_ECFP4 fingerprints and XGBoost.This model is the optimal individual model.
   - Model 2A-M generates predictions using RXNFP embeddings and XGBoost.
   - Model 3B-M generates predictions using DRFP fingerprints and a deep neural network.
   - Model 4B-M generates predictions directly from reaction SMILES using a ChemMLM-based model.

3. **Consensus prediction**
   - The four predicted probabilities are averaged.
   - The averaged probability is used as the final prediction probability.
   - A threshold of 0.5 is used to assign the final feasibility label.

## Prediction
﻿
The complete prediction workflow is provided in:
﻿
**`prediction_workflow.ipynb`**
﻿
The notebook demonstrates the complete process from reaction preprocessing to consensus prediction.
