# Multi-Armed-Bandits

## REQUIREMENTS
```bash
* shap==0.39.0
* scikit-learn==0.24.2
* matplotlib==3.4.3 
* numpy==1.20.3 
* xgboost==1.4.2
* lightgbm==3.2.1 
* mkl_random==1.2.2
* jsonschema==3.2.0
* scipy==1.7.1 
* sortedcollections==1.2.1
```

Function of interest: main().

Update config_file.json to start a policy:

```bash
* "file_path": path to the dataset (raw format).
* "limit_size": the maximum events to process.
* "policy_name": the exact name of the policy.
* "args":
		* "policy_args": policy arguments.
		* "model_args": model arguments (compatible with sklearn, xgboost, lightgbm libraries).
```
