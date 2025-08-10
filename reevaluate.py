import os
import re
import torch
from init_parameter import init_model
from dataloader import Data
from model import CLBert, BertForModel
from GCDLLMs import ModelManager


def parse_metadata_from_filename(filename: str):
    """Parse dataset and key ratios from checkpoint filename, if present.
    Expected patterns like:
    <dataset>_known_cls_ratio_<known>_labeled_ratio_<labeled>[_weight_cluster_instance_cl_<wcl>][_evaluation_epoch_<epoch>].pt
    """
    base = os.path.basename(filename)
    # Strip extension
    if base.endswith('.pt'):
        base = base[:-3]
    pattern = (
        r"^(?P<dataset>[A-Za-z0-9_]+)"
        r"_known_cls_ratio_(?P<known>[0-9.]+)"
        r"_labeled_ratio_(?P<labeled>[0-9.]+)"
        r"(?:_weight_cluster_instance_cl_(?P<wcl>[0-9.]+))?"
        r"(?:_evaluation_epoch_(?P<epoch>-?\d+))?"
        r"$"
    )
    m = re.match(pattern, base)
    if not m:
        return None
    md = m.groupdict()
    out = {
        "dataset": md.get("dataset"),
        "known_cls_ratio": float(md["known"]) if md.get("known") else None,
        "labeled_ratio": float(md["labeled"]) if md.get("labeled") else None,
    }
    if md.get("wcl") is not None:
        out["weight_cluster_instance_cl"] = float(md["wcl"])
    if md.get("epoch") is not None:
        out["evaluation_epoch"] = int(md["epoch"])
    return out


def build_model_from_checkpoint(args, checkpoint_path: str, device: torch.device) -> CLBert:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", args.bert_model)
    num_labels = ckpt.get("num_labels")
    feat_dim = ckpt.get("feat_dim", args.feat_dim)
    architecture = ckpt.get("architecture")
    if architecture is not None:
        args.architecture = architecture

    model = CLBert(args, model_name, device=device, num_labels=num_labels, feat_dim=feat_dim)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def main():
    parser = init_model()
    # Relax required constraints to allow inferring from filename
    for action in parser._actions:
        if getattr(action, "dest", None) == "dataset":
            action.required = False
        if getattr(action, "dest", None) == "known_cls_ratio":
            action.required = False
        if getattr(action, "dest", None) == "seed":
            action.required = False
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to a saved checkpoint .pt file")
    args = parser.parse_args()

    # Ensure fields used by save_results exist
    if not hasattr(args, "evaluation_epoch"):
        args.evaluation_epoch = -1

    # Infer missing fields from checkpoint filename
    inferred = parse_metadata_from_filename(args.checkpoint_path)
    if args.dataset is None:
        if inferred and inferred.get("dataset"):
            args.dataset = inferred["dataset"]
        else:
            raise ValueError("--dataset not provided and cannot be inferred from checkpoint filename.")
    if inferred:
        if inferred.get("known_cls_ratio") is not None:
            args.known_cls_ratio = inferred["known_cls_ratio"]
        if inferred.get("labeled_ratio") is not None:
            args.labeled_ratio = inferred["labeled_ratio"]
        if inferred.get("weight_cluster_instance_cl") is not None:
            args.weight_cluster_instance_cl = inferred["weight_cluster_instance_cl"]
        if inferred.get("evaluation_epoch") is not None:
            args.evaluation_epoch = inferred["evaluation_epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data/eval loaders with current args
    data = Data(args)

    # Build manager with a dummy pretrained backbone to bypass loading paths
    dummy_pretrained = BertForModel(args.bert_model, num_labels=data.n_known_cls if hasattr(data, 'n_known_cls') else data.num_labels)
    manager = ModelManager(args, data, dummy_pretrained)

    # Load the saved full model directly from the file path
    manager.model = build_model_from_checkpoint(args, args.checkpoint_path, device)
    manager.num_labels = manager.model.num_labels

    print("Evaluation begin (reevaluate_simple.py)...")
    manager.evaluation(args, data, save_results=True, plot_cm=False)
    print("Evaluation finished (reevaluate_simple.py).")


if __name__ == "__main__":
    main()


