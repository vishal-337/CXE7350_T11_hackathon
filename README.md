# CXE7350_T11_hackathon

## `Model_Attempts/`


| File                                  | Description                                                                                                                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Hackathon_script.ipynb`              | ESM-2 embeddings, sklearn gradient boosting, ridge, 0.6/0.4 blend, KMeans for queries                                                                                                                                            |
| `Hackathon_script (2).ipynb`          | Same as above, plus PyTorch MLP on ESM features                                                                                                                                                                                  |
| `The_Hackathon_predictions (3).ipynb` | Gradient boosting, ridge, XGBoost, extra trees, hist gradient boosting, PyTorch MLP                                                                                                                                              |
| `MLB26_Hack.ipynb`                    | XGBoost with `objective='rank:pairwise'` (pairwise ranking on labels) instead of `reg:squarederror` like in `The_Hackathon_predictions (3).ipynb`                                                                                |
| `nn_q3.py`                            | Optional PyTorch MLP on the same hand-crafted mutation encoding as the final ensemble (for comparison)                                                                                                                           |
| `disagreement_query.py`               | Final pipeline: five sklearn models (gradient boosting, hist gradient boosting, ridge, extra trees, random forest), disagreement-based query selection with binned sequence diversity, ensemble test predictions and `top10.txt` |


The main file to run end-to-end is `**Model_Attempts/disagreement_query.py**`.

From the repository root, install dependencies :

```bash
pip install -r requirements.txt
```

Then run the script from `Model_Attempts` (paths assume the repo root is one level up). On many systems only `python3` is available, not `python`:

```bash
cd Model_Attempts
python3 disagreement_query.py
```

It reads `Hackathon_data/` (including `train.csv`, `test.csv`, `queryGroundTruth1.csv`, `sequence.fasta`) plus `queries/query2/q2_results.csv` and `queries/query3/q3_results.csv`. It writes `queries/query3/query3.txt`, `predictions/ensemble/predictions.csv` (main output), and `predictions/ensemble/top10.txt`.