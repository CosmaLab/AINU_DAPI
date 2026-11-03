### REMEMBER - to analyze log files output from only test, copy the "running parameters"
### of the model you tested the dataset on into the logfile of the "only test"

import argparse
import ast
import os
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from captum.attr import Occlusion
from captum.attr import visualization as viz
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from torchvision import models

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 7})


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_logfile(log_file):
    obj = None
    test_set = None
    with open(log_file) as f:
        for line in f:
            if "Running parameters" in line:
                temp = line.replace("'", '"')
                obj = ast.literal_eval(temp.split("Running parameters: ")[1])
            elif "Test set: " in line:
                test_set = ast.literal_eval(line.split("Test set: ")[1])
    if obj is None:
        raise ValueError("No 'Running parameters' line found in log file.")
    if test_set is None:
        raise ValueError("No 'Test set:' line found in log file.")
    return obj, test_set


def main():
    parser = argparse.ArgumentParser(
        description="Occlusion attribution for any class, parsed from a training log file."
    )
    parser.add_argument("--log_file", type=str, required=True,
                        help="Path to the log file of the images")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to output folder for occlusion results")
    parser.add_argument("--savenpy", action="store_true",
                        help="Save occlusion maps as .npy files")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Classes to analyze (default: all classes in the dataset)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference (default: 16)")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    obj, test_set = parse_logfile(args.log_file)

    # Resolve label file
    if obj.get("only_test"):
        label_path = obj["test_data_path"]
    else:
        label_path = obj["data"] + obj["filelist"]

    lista = pd.read_csv(label_path, sep="\t", header=0)
    label_names = sorted(lista.condition.unique())
    labels = {name: i for i, name in enumerate(label_names)}
    print("Labels:", labels)

    get_test_set_labels = [os.path.splitext(j)[0] for j in test_set]
    test_labels = lista[lista["File"].isin(get_test_set_labels)]
    print(f"Test images: {len(test_labels)}")

    # Load model
    augmentation_path = obj["augdir"] + "/test/" + obj["augment"] + ".json"
    if obj.get("only_test"):
        model_path = obj["test_model_path"]
    else:
        model_path = obj["outdir"] + "/" + obj["name"] + "_model_best.pth.tar"

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, obj["classes"])
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)["state_dict"]
    )
    model = model.to(device)
    model.eval()

    transform = A.load(augmentation_path)
    label_idx = {i: name for name, i in labels.items()}  # reverse map: int -> class name
    print(label_idx)
    images_dir = obj.get("test_images_dir") or obj.get("data")
    classes_to_analyze = args.classes if args.classes else label_names

    # --- Phase 1: batch inference over all test images, save predictions CSV ---
    print("\nRunning batch inference...")
    all_predictions = []
    for class_name in label_names:
        class_files = test_labels[test_labels["condition"] == class_name]["File"].to_numpy()
        for batch_start in range(0, len(class_files), args.batch_size):
            batch_files = class_files[batch_start:batch_start + args.batch_size]
            batch_tensors, batch_names = [], []
            for img_name in batch_files:
                img_path = Path(images_dir) / f"{img_name}.png"
                if not img_path.exists():
                    print(f"  Warning: image not found: {img_path}")
                    continue
                img = Image.open(img_path).convert("RGB")
                transformed = transform(image=np.asarray(img))
                batch_tensors.append(transformed["image"] / 255.0)
                batch_names.append(img_name)

            if not batch_tensors:
                continue

            batch_input = torch.stack(batch_tensors).float().to(device)
            with torch.no_grad():
                batch_output = F.softmax(model(batch_input), dim=1)
                batch_scores, batch_pred_idx = torch.topk(batch_output, 1)

            for i, img_name in enumerate(batch_names):
                pred_idx_val = batch_pred_idx[i].squeeze().item()
                predicted_label = label_idx[pred_idx_val]
                score = batch_scores[i].squeeze().item()
                all_predictions.append({
                    "image_name": img_name,
                    "true_class": class_name,
                    "predicted_class": predicted_label,
                    "prediction_score": score,
                    "correct": predicted_label == class_name,
                })

    predictions_df = pd.DataFrame(all_predictions)
    csv_path = out_path / "predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    print(predictions_df.groupby("true_class")["correct"].agg(["sum", "count"]).rename(
        columns={"sum": "correct", "count": "total"}
    ))

    # --- Phase 2: occlusion for correctly predicted images ---
    occlusion = Occlusion(model)

    for class_name in classes_to_analyze:
        if class_name not in labels:
            print(f"\nClass '{class_name}' not in labels, skipping.")
            continue

        class_idx = labels[class_name]
        correct_preds = [
            p for p in all_predictions
            if p["true_class"] == class_name and p["correct"]
        ]
        print(f"\nOcclusion for class '{class_name}': "
              f"{len(correct_preds)} correctly predicted images")

        if not correct_preds:
            print("  No correctly predicted images, skipping.")
            continue

        pdf_path = out_path / f"{class_name}_occlusion.pdf"
        with PdfPages(pdf_path) as pdf:
            for pred in correct_preds:
                name = pred["image_name"]
                score = pred["prediction_score"]
                predicted_label = pred["predicted_class"]

                img_path = Path(images_dir) / f"{name}.png"
                img = Image.open(img_path).convert("RGB")
                transformed = transform(image=np.asarray(img))
                img_tensor = transformed["image"] / 255.0
                input_tensor = img_tensor.unsqueeze(0).float().to(device)

                print(f"  {name} -> {predicted_label} ({score:.4f})")

                attributions_occ = occlusion.attribute(
                    input_tensor,
                    strides=(3, 4, 4),
                    target=1,
                    sliding_window_shapes=(3, 8, 8),
                    baselines=0,
                    perturbations_per_eval=64,
                )
                attribution_mask = attributions_occ.squeeze().cpu().detach().numpy()

                if args.savenpy:
                    np.save(out_path / f"{name}_{class_name}_AML_occlusion.npy", attribution_mask)

                img_hwc = np.transpose(img_tensor.squeeze().cpu().numpy(), (1, 2, 0))
                attr_hwc = np.transpose(attribution_mask, (1, 2, 0))

                # Use visualize_image_attr directly to avoid Captum 0.8.0 bug
                # where visualize_image_attr_multiple leaves subplots blank.
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                viz.visualize_image_attr(
                    attr_hwc, img_hwc,
                    method="original_image",
                    sign="all",
                    plt_fig_axis=(fig, axes[0]),
                    use_pyplot=False,
                    title="Original",
                )
                viz.visualize_image_attr(
                    attr_hwc, img_hwc,
                    method="heat_map",
                    sign="all",
                    plt_fig_axis=(fig, axes[1]),
                    show_colorbar=True,
                    use_pyplot=False,
                    title="Occlusion Heatmap",
                )
                viz.visualize_image_attr(
                    attr_hwc, img_hwc,
                    method="blended_heat_map",
                    sign="all",
                    plt_fig_axis=(fig, axes[2]),
                    show_colorbar=True,
                    alpha_overlay=0.5,
                    use_pyplot=False,
                    title="Overlay",
                )
                fig.suptitle(f"{name}\nPredicted: {predicted_label} (score: {score:.4f})")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close("all")

        print(f"  Saved: {pdf_path}")


if __name__ == "__main__":
    main()
