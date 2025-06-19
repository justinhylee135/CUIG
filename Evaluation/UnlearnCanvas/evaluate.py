# Standard Library
import os
import argparse
from PIL import Image
import ast
import json
from pathlib import Path

# Third Party
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
import numpy as np
import torch
torch.hub.set_dir("cache")
from torchvision import transforms
import timm
from tqdm import tqdm

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory for images folders",required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to output results.",required=True)
    parser.add_argument("--seed", type=int, nargs="+", help="Seeds to use for sampling",default=[188, 288, 588, 688, 888])
    parser.add_argument("--classifier_dir", type=str, help="Directory that holds style and object classifier",required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--unlearn", type=str, help="Set of concepts that were unlearned", default=None)
    parser.add_argument("--retain", type=str, help="Set of in-domain concepts to retain", default=None)
    parser.add_argument("--cross_retain", type=str, help="Set of cross-domain concepts to retain", default=None)
    args = parser.parse_args()

    # Set image input and results output directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device to use
    torch.cuda.set_device(args.device) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Constants
    theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
                    "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
                    "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
                    "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
                    "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
                    "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
                    "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
                    "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]
    object_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                    "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                    "Trees", "Waterfalls"]
    
    # Default to all themes and classes
    theme_available_subset = theme_available
    object_available_subset = object_available
    
    # Check if concept sets were specified
    if args.unlearn is not None and args.retain is not None and args.cross_retain is not None:
        # Parse the unlearn, retain, and cross_retain arguments
        unlearn = ast.literal_eval(args.unlearn)
        retain = ast.literal_eval(args.retain)
        cross_retain = ast.literal_eval(args.cross_retain)
        print(f"Using Unlearn: {unlearn}")
        print(f"Using Retain: {retain}")
        print(f"Using Cross Retain: {cross_retain}")

        # Check first concept in unlearn to determine if it's a theme or object
        is_theme_unlearn = unlearn[0] in theme_available
        
        # Set available subsets based on the unlearn concept type
        if is_theme_unlearn:
            theme_available_subset = unlearn + retain
            object_available_subset = cross_retain
        else:
            theme_available_subset = cross_retain
            object_available_subset = unlearn + retain
    
    # Set image transformation
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Seeds to iterate over for generation
    seed_list = args.seed
    
    # Perform evaluation for style then object
    TASKS_LIST = ["style", "object"]
    for TASK in TASKS_LIST:
        # Set output path for results json file
        output_path = os.path.join(output_dir, f"{TASK}_results.json")

        # Set pretrained classifier
        model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
        num_classes = len(theme_available) if TASK == "style" else len(object_available)
        model.head = torch.nn.Linear(1024, num_classes).to(device)
        
        # Load task specific classifier checkpoint
        classifier_path = Path(args.classifier_dir) / f"{TASK}_classifier.pth"
        model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=False)["model_state_dict"])
        model.eval()
        
        # Initialize results json format
        if TASK == "style":
            results = {
                "input_dir": args.input_dir,
                "acc": {theme: 0.0 for theme in theme_available_subset},
                "misclassified": {theme: {other_theme: 0 for other_theme in theme_available} for theme in theme_available_subset},
                "image_specific": {},
                "eval_count": {theme: 0 for theme in theme_available_subset},
                "pred_loss": {theme: 0.0 for theme in theme_available_subset},
                "loss": {theme: 0.0 for theme in theme_available_subset}
            }
        else:
            results = {
                "input_dir": args.input_dir,
                "acc": {obj: 0.0 for obj in object_available_subset},
                "misclassified": {obj: {other_object: 0 for other_object in object_available} for obj in object_available_subset},
                "image_specific": {},
                "eval_count": {obj: 0 for obj in object_available_subset},
                "pred_loss": {obj: 0.0 for obj in object_available_subset},
                "loss": {obj: 0.0 for obj in object_available_subset}
            }

        # Evaluation loop with tqdm progress bar
        total_steps = len(theme_available_subset) * len(seed_list) * len(object_available_subset)
        with tqdm(total=total_steps, desc=f"Evaluating {TASK}", unit="img") as pbar:
            # Style evaluation
            if TASK == "style":
                for idx, test_theme in enumerate(theme_available_subset):
                    # Ground truth corresponds to the list index
                    theme_label = theme_available.index(test_theme)
                    for seed in seed_list:
                        for test_object in object_available_subset:
                            # Path to where image should be
                            img_path = os.path.join(input_dir, f"{test_theme}_{test_object}_seed{seed}.jpg")

                            # Skip if image doesn't exist
                            if not os.path.exists(img_path):
                                print(f"Warning: Image {img_path} not found. Skipping...")
                                pbar.update(1)
                                continue
                            
                            # Load image
                            image = Image.open(img_path)
                            target_image = image_transform(image).unsqueeze(0).to(device)

                            # Run classifier
                            with torch.no_grad():
                                # Class Predictions
                                res = model(target_image)
                                
                                # Get label with highest score
                                pred_label = torch.argmax(res)

                                # Load ground truth into tensor
                                label = torch.tensor([theme_label]).to(device)
                                
                                # Calculate cross-entropy loss
                                loss = torch.nn.functional.cross_entropy(res, label)

                                # Softmax the prediction and get loss
                                res_softmax = torch.nn.functional.softmax(res, dim=1)
                                pred_loss = res_softmax[0][theme_label]
                                
                                # Check if prediction is correct
                                pred_success = (pred_label == theme_label).sum()

                            # Update results
                            results["loss"][test_theme] += loss.item()
                            results["pred_loss"][test_theme] += pred_loss.item()
                            results["acc"][test_theme] += pred_success.item()
                            results["eval_count"][test_theme] += 1
                            misclassified_as = theme_available[pred_label.item()]
                            results["misclassified"][test_theme][misclassified_as] += 1
                            short_img_path = os.path.splitext(img_path.split("images/")[1])[0]
                            results["image_specific"][short_img_path] = theme_available[pred_label.item()]
                            pbar.update(1)
                                                
            # Object evaluation                            
            elif TASK == "object":
                for test_theme in theme_available_subset:
                    for seed in seed_list:
                        for idx, test_object in enumerate(object_available_subset):
                            # Get ground truth from the list index
                            object_label = object_available.index(test_object)
                            
                            # Path to where image should be
                            img_path = os.path.join(input_dir, f"{test_theme}_{test_object}_seed{seed}.jpg")
                            
                            # Skip if image doesn't exist
                            if not os.path.exists(img_path):
                                print(f"Warning: Image {img_path} not found. Skipping...")
                                pbar.update(1)
                                continue
                            
                            # Load image
                            image = Image.open(img_path)
                            target_image = image_transform(image).unsqueeze(0).to(device)

                            # Run classifier
                            with torch.no_grad():
                                # Class Predictions
                                res = model(target_image)
                                pred_label = torch.argmax(res)

                                # Load ground truth into tensor and get cross-entropy loss
                                label = torch.tensor([object_label]).to(device)
                                loss = torch.nn.functional.cross_entropy(res, label)

                                # Softmax the prediction and get loss
                                res_softmax = torch.nn.functional.softmax(res, dim=1)
                                pred_loss = res_softmax[0][object_label]
                                
                                # Check if prediction is correct
                                pred_success = (pred_label == object_label).sum()
                            
                            # Update results
                            results["loss"][test_object] += loss.item()
                            results["pred_loss"][test_object] += pred_loss.item()
                            results["acc"][test_object] += pred_success.item()
                            results["eval_count"][test_object] += 1
                            misclassified_as = object_available[pred_label.item()]
                            results["misclassified"][test_object][misclassified_as] += 1
                            short_img_path = os.path.splitext(img_path.split("images/")[1])[0]
                            results["image_specific"][short_img_path] = object_available[pred_label.item()]
                            pbar.update(1)
                
        # Normalize results (exclude images that weren't found)            
        for k in results["acc"]:
            count = results["eval_count"][k]
            results["acc"][k] = results["acc"][k] / count if count > 0 else 0.0
            results["acc"][k] = round(100 * results["acc"][k], 2)

        # Remove counts that are 0 from misclassified
        for concept in results["misclassified"]:
            for other_concept in list(results["misclassified"][concept].keys()):
                if results["misclassified"][concept][other_concept] == 0:
                    del results["misclassified"][concept][other_concept]
                    
        # Save results to JSON file
        with open(output_path, 'w') as f:
            # Convert tensors to floats for JSON serialization
            serializable_results = {
                k: {sk: (float(sv) if isinstance(sv, torch.Tensor) else sv)
                    for sk, sv in v.items()} if isinstance(v, dict) else v
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=4)
        
        # Store results in the appropriate variable for later use
        if TASK == "style":
            results_style = results
        else:
            results_object = results
    
    # Choose result dictionaries based on the unlearn concept type
    primary_results = results_style if is_theme_unlearn else results_object
    secondary_results = results_object if is_theme_unlearn else results_style

    # Log individual unlearn accuracies.
    for item in unlearn:
        accuracy = 100.0 - primary_results["acc"][item]
        print(f"Unlearn accuracy for {item}: {accuracy:.2f}%")

    # Compute average accuracies.
    unlearn_accuracy_avg = 100.0 - np.mean([primary_results["acc"][item] for item in unlearn])
    retain_accuracy = np.mean([primary_results["acc"][item] for item in retain])
    cross_retain_accuracy = np.mean([secondary_results["acc"][item] for item in cross_retain])

    # Create summary json
    summary = {
        "UA": round(unlearn_accuracy_avg, 2),
        "IRA": round(retain_accuracy, 2),
        "CRA": round(cross_retain_accuracy, 2),
        "unlearn": {},
        "retain": {},
        "cross_retain": {}
    }

    # Add scores to summary
    for concept in unlearn:
        summary["unlearn"][concept] = (100.0 - primary_results["acc"][concept])
    for concept in retain:
        summary["retain"][concept] = primary_results["acc"][concept]
    for concept in cross_retain:
        summary["cross_retain"][concept] = secondary_results["acc"][concept]
    
    # Save summary to JSON file
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    # Print summary
    print(f"Unlearn accuracy (average): {unlearn_accuracy_avg:.2f}%")
    print(f"Retain accuracy (average): {retain_accuracy:.2f}%")
    print(f"Cross retain accuracy (average): {cross_retain_accuracy:.2f}%")
    
    # ---------------------------
    # Write Excel output with metrics
    # ---------------------------
    
    # These are the concept subsets we used for our paper (change as needed)
    xlsx_all_unlearn_themes = ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm", "Ukiyoe", "Van_Gogh", 
                               "Neon_Lines", "Picasso", "On_Fire", "Magic_Cube", "Winter", "Vibrant_Flow"]
    xlsx_all_retain_themes = ["Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", 
                               "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]
    xlsx_all_unlearn_objects = ["Bears", "Birds", "Cats", "Dogs", "Fishes", "Frogs", "Jellyfish", "Rabbits", "Sandwiches", "Statues", "Towers", "Waterfalls"]
    xlsx_all_retain_objects = ["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]
    
    # We determine the full sets to output as a table later on
    if unlearn[0] in xlsx_all_unlearn_themes:
        full_unlearn = xlsx_all_unlearn_themes
        full_retain = xlsx_all_retain_themes
        full_cross_retain = xlsx_all_retain_objects
    else:
        full_unlearn = xlsx_all_unlearn_objects
        full_retain = xlsx_all_retain_objects
        full_cross_retain = xlsx_all_retain_themes
        
    # Create a new Excel workbook and sheet
    run_name = f"{unlearn[0]}" if len(unlearn) == 1 else f"Thru{unlearn[-1]}"    
    xlsx_path = os.path.join(output_dir, f"{run_name}_table.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Metrics"

    # Excel header: first column "Metric", then one column per style in the unlearn list.
    header = ["Metric"] + [run_name]
    ws.append(header)

    # Store rows
    rows = []

    # Function to store values correctly as percentages (e.g., 85.0 → 0.85)
    def to_percentage(value):
        return round(value / 100, 4)  # Convert to fraction for Excel (85.0 → 0.85)

    # Placeholder value if a concept is not present
    placeholder = "x"

    # UA: for each unlearn style, compute UA = (1 - primary_results["acc"][style]) * 100
    for style in full_unlearn:
        if style in unlearn:
            ua = to_percentage((1 - primary_results["acc"][style]) * 100)
            rows.append([f"UA: {style}", ua])
        else:
            rows.append([f"UA: {style}", placeholder])
            
    # IRA: for each corresponding retain concept, compute IRA = primary_results["acc"][concept] * 100
    for concept in full_retain:
        if concept in retain:
            ira = to_percentage(primary_results["acc"][concept] * 100)
            rows.append([f"IRA: {concept}", ira])
        else:
            rows.append([f"IRA: {concept}", placeholder])

    # CRA: for each corresponding cross_retain concept, compute CRA = secondary_results["acc"][concept] * 100
    for concept in full_cross_retain:
        if concept in cross_retain:
            cra = to_percentage(secondary_results["acc"][concept] * 100)
            rows.append([f"CRA: {concept}", cra])
        else:
            rows.append([f"CRA: {concept}", placeholder])

    # Append computed average rows
    avg_ua = to_percentage(unlearn_accuracy_avg * 100)
    avg_ira = to_percentage(retain_accuracy * 100)
    avg_cra = to_percentage(cross_retain_accuracy * 100)
    rows.append(["Avg UA", avg_ua])
    rows.append(["Avg IRA", avg_ira])
    rows.append(["Avg CRA", avg_cra])

    # Write rows to Excel
    for row in rows:
        ws.append(row)

    # Define fonts
    roboto_font = Font(name="Roboto", size=12)
    bold_roboto_font = Font(name="Roboto", size=12, bold=True)

    # Define alignment separately
    default_align = Alignment(vertical="center", wrap_text=False)
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=False)

    # Set table dimensions
    num_rows = len(rows) + 1  # +1 because Excel is 1-indexed
    num_cols = len(header)  # Get number of columns

    # Apply font, vertical alignment, and text wrapping
    for row_idx in range(1, num_rows + 1):  # Loop through all rows
        for col_idx in range(1, num_cols + 1):  # Loop through all columns
            cell = ws[f"{get_column_letter(col_idx)}{row_idx}"]
            cell.font = roboto_font  # Apply Roboto font
            cell.alignment = default_align  # Apply vertical align + wrap text

            # Center align only numeric values (column B and beyond, skipping first row)
            if col_idx > 1 and row_idx > 1:
                cell.alignment = center_align

    # Bold the last three rows
    for row_idx in range(num_rows - 2, num_rows + 1):
        for col_idx in range(1, num_cols + 1):
            ws[f"{get_column_letter(col_idx)}{row_idx}"].font = bold_roboto_font

    # Apply percentage format to all numeric cells in column B and beyond
    for col_idx in range(2, num_cols + 1):  # Start from column B
        col_letter = get_column_letter(col_idx)
        for row_idx in range(2, num_rows + 1):  # Start from row 2 (skip header)
            if isinstance(ws[f"{col_letter}{row_idx}"].value, (int, float)):  # Only apply to numbers
                ws[f"{col_letter}{row_idx}"].number_format = "0.00%"

    # Save Excel file
    wb.save(xlsx_path)
    print(f"Excel file saved at: {xlsx_path}")