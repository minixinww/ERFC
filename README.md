# ERFC
We developed an XGBoost model (Model_1A_4) based on C_ECFP4 and a DNN model (Model_2B_2) based on RXNFP to classify the feasibility of enzymatic reactions. These models enable efficient and rapid screening of large-scale biosynthetic reactions.
The project includes five directories:  
- `models/` stores the trained Model_1A_4 and Model_2B_2 model files;  
- `notebooks/` contains Python scripts implementing the complete prediction workflow: from reaction representation to model inference and result output;  
- `example/` provides standardized example reactions, enabling users to quickly test and verify the process;  
- `results/` saves the generated c_ECFP4 or RXNFP fingerprints, as well as the predicted classification labels and probabilities, for subsequent analysis.
