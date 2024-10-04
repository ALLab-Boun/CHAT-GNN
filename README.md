# CHAT-GNN

This is the official repository of "Channel-Attentive Graph Neural Networks, ICDM 2024".

## Setup

Tested with Python 3.9 and 3.10.

```bash
python -m venv chatgnn_env
source chatgnn_env/bin/activate
pip install -r requirements.txt
# pyg dependencies
pip install pyg_lib==0.4.0 \
            torch_scatter==2.1.1 \
            torch_sparse==0.6.17 \
            torch_cluster==1.6.3 \
            torch_spline_conv==1.2.2 \
            -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

## Reproducing the results

An example command for a single dataset:

```bash
python run.py --config-path output_hyp/chatgnn_minesweeper_best.json
```

This will train a CHAT-GNN on minesweeper dataset with the best hyperparameters found in hyperparameter search. The test results on each split and the average test results will be output to the screen. You can run another experiment by changing the dataset name.

Note: It's hard to fully reproduce the results due to the non-deterministic nature of GPU and the specific operations done in torch geometric (see [link](https://github.com/pyg-team/pytorch_geometric/issues/92#issuecomment-472332656)). However, the results should be close to the paper.

## Implementation

We implemented our model by creating a module for our proposed message-passing layer. It is in layers/chatconv.py. Our whole architecture is implemented in model.py.

We leveraged the repositories [AERO-GNN](https://github.com/syleeheal/AERO-GNN), [Ordered GNN](https://github.com/LUMIA-Group/OrderedGNN) for some of the baseline implementations.


## Hyperparameter search

We run 75 iterations of Bayesian optimization for each model and dataset using wandb. We evaluate the models using the average validation accuracy of the first three splits. Finally, we select the hyperparameter settings with the highest validation accuracy and re-train the models with the same configuration using the ten dataset splits.

The exceptions here are [AERO-GNN](https://github.com/syleeheal/AERO-GNN) and [Ordered GNN](https://github.com/LUMIA-Group/OrderedGNN). We run both models with the reported best hyperparameters instructed by the authors for the common benchmark datasets. When the performance on some of the benchmark datasets is missing in the baseline studies, we train the baseline models from scratch by running hyperparameter optimization considering the suggested hyperparameter setting by the authors, dataset scale, and our computational resources.

### Search spaces

You can examine all the hyperparameter search spaces in the folder "configs/hyp_search".

## Hardware

We use NVIDIA Tesla T4 (16GiB) and NVIDIA Tesla V100 (16GiB) as our GPU hardware.

## Supplementary Materials

Due to page limits, we have included the proofs and the model architecture diagram in the [supplementary_materials](supplementary_materials) folder.


## Bibtex

(Soon...)

<!-- ```
@inproceedings{karabulut2024channel,
  author={Karabulut, Tuğrul Hasan and Baytaş, İnci M.},
  booktitle={2024 IEEE International Conference on Data Mining (ICDM)}, 
  title={Channel-Attentive Graph Neural Networks}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}
}
``` -->