# CXE7350_T11_hackathon

## `Model Attemps/`

| Notebook | Models |
| --- | --- |
| `Hackathon_script.ipynb` | ESM-2 embeddings, sklearn gradient boosting, ridge, 0.6/0.4 blend, KMeans for queries |
| `Hackathon_script (2).ipynb` | Same as above, plus PyTorch MLP on ESM features |
| `The_Hackathon_predictions (3).ipynb` | Gradient boosting, ridge, XGBoost, extra trees, hist gradient boosting, PyTorch MLP |
| `MLB26_Hack.ipynb` | XGBoost with `objective='rank:pairwise'` (pairwise ranking on labels) instead of `reg:squarederror` like in `The_Hackathon_predictions (3).ipynb` |

Most recent version: `MLB26_Hack.ipynb`.\n
Most recent prediction file:`predictions/XGBoost_rank:pairwise/predictions.csv`.
