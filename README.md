# ERFC
We developed an XGBoost model Model 1A-M based on c_ECFP4 to classify the feasibility of enzymatic reactions, which can handle both stereochemistry-preserving and stereochemistry-agnostic reactions. These models enable efficient and rapid screening of large-scale biosynthetic reactions.
The project includes five directories:  
- `model/` stores the trained Model 1A-M model file;  
- `notebooks/` contains Python scripts implementing the complete prediction workflow: from reaction representation to model inference and result output;  
- `example/` provides standardized example reactions, enabling users to quickly test and verify the process;  
- `results/` saves the generated c_ECFP4 fingerprints, as well as the predicted classification labels and probabilities, for subsequent analysis.
