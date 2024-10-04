import torch
from sklearn.metrics import roc_auc_score

from util import dirichlet_energy, mad_stats


@torch.no_grad()
def evaluate(
    model,
    dataset,
    split=0,
    loss_func=None,
    calc_mad_gap=False,
    calc_dirichlet_energy=False,
    eval_metric="acc",
):
    graph = dataset[0]
    x = graph.x
    y = graph.y
    edge_index = graph.edge_index
    if graph.train_mask.ndim > 1:
        train_mask = graph.train_mask[:, split]
        val_mask = graph.val_mask[:, split]
        test_mask = graph.test_mask[:, split]
    else:
        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask

    model.eval()
    model.activate_store_emb()
    pred = model(x, edge_index)

    if loss_func:
        if dataset.num_classes == 2:
            pred = pred.squeeze()
            y = y.to(torch.float32)
        train_loss = loss_func(pred[train_mask], y[train_mask]).detach().item()
        val_loss = loss_func(pred[val_mask], y[val_mask]).detach().item()
        test_loss = loss_func(pred[test_mask], y[test_mask]).detach().item()
    else:
        train_loss = None
        val_loss = None
        test_loss = None

    if eval_metric == "acc":
        if dataset.num_classes == 2:
            pred_cls = (pred >= 0.5).to(torch.float32)
        else:
            pred_cls = pred.argmax(1)
        train_eval = (pred_cls[train_mask] == y[train_mask]).float().mean().item()
        val_eval = (pred_cls[val_mask] == y[val_mask]).float().mean().item()
        test_eval = (pred_cls[test_mask] == y[test_mask]).float().mean().item()
    elif eval_metric == "auc":
        train_eval = roc_auc_score(y[train_mask].cpu(), pred[train_mask].cpu())
        val_eval = roc_auc_score(y[val_mask].cpu(), pred[val_mask].cpu())
        test_eval = roc_auc_score(y[test_mask].cpu(), pred[test_mask].cpu())

    eval_dict = {
        "train_loss": round(train_loss, 3) if loss_func else None,
        "train_eval": round(train_eval, 3),
        "test_loss": round(test_loss, 3) if loss_func else None,
        "test_eval": round(test_eval, 3),
        "val_loss": round(val_loss, 3) if loss_func else None,
        "val_eval": round(val_eval, 3),
        "eval_metric": eval_metric,
    }

    if calc_mad_gap:
        mad_gap_val, mad_ratio_val = mad_stats(model.x_emb, edge_index)
        eval_dict["mad_gap_val"] = round(mad_gap_val, 3)
        eval_dict["mad_ratio_val"] = round(mad_ratio_val, 3)

    if calc_dirichlet_energy:
        eval_dict["dirichlet_energy"] = dirichlet_energy(model.x_emb, edge_index)

    model.deactivate_store_emb()

    return eval_dict
