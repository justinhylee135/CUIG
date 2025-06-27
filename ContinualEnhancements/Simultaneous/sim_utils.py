# Standard Library
import json
import os
from PIL import Image
from pathlib import Path

# Third Party
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from tqdm import tqdm
from torchvision import transforms
import timm

def sample_and_evaluate_ua(pipeline, concept_type, iteration, model_save_path, prompt_list, prompt, device, classifier_dir):
    # Lists of available theme and classes
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

    # Set up sampler and folders
    output_dir = os.path.join(Path(model_save_path).parent, f"logs/log_{iteration}")
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")  
    os.makedirs(img_dir, exist_ok=True)
    print(f"Saving images to: {img_dir}")

    # Define concepts and seeds to generate
    seed_list = [188, 288, 588, 688, 888]
    object_available_subset = []
    theme_available_subset = []
    for prompt in prompt_list:
        concept = prompt.replace('An image of ', '')
        concept = concept.replace(' Style', '')
        concept = concept.replace(' ', '_')
        if concept in theme_available:
            theme_available_subset.append(concept)
        elif concept in object_available:
            object_available_subset.append(concept)
        else:
            raise ValueError(f"Concept '{concept}' not found in available themes or classes.")    
    unlearn_subset = object_available_subset + theme_available_subset
    print(f"Unlearn Subset: {unlearn_subset}")
    
    # Choose cross-domain concept for sampling
    if concept_type == "style":
        retention_themes = []
        retention_classes = ["Architectures", "Butterfly", "Flame"]
    elif concept_type == "object":
        retention_themes = ["Blossom_Season", "Rust", "Crayon"]
        retention_classes = []
    elif concept_type == "interleave":
        retention_themes = ["Blossom_Season", "Rust", "Crayon"]
        retention_classes = ["Architectures", "Butterfly", "Flame"]
    theme_available_subset.extend(retention_themes)
    object_available_subset.extend(retention_classes)
    print(f"Sampling:")
    print(f"\tClasses subset: {object_available_subset}")
    print(f"\tThemes subset: {theme_available_subset}")
    
    # Switch to evaluation mode (we'll switch back later)
    was_training = pipeline.unet.training
    pipeline.unet.eval()
    cfg_text = 9.0
    H = 512
    W = 512
    steps = 100
    
    # Sample
    total_iterations = len(seed_list) * len(theme_available_subset) * len(object_available_subset)
    with tqdm(total=total_iterations, desc="Generating images", unit="image") as pbar:
        for seed in seed_list:
            seed_everything(seed)
            for test_theme in theme_available_subset:
                for object_class in object_available_subset:
                    # Define save path
                    img_save_path = os.path.join(img_dir, f"{test_theme}_{object_class}_seed{seed}.jpg")
                    if os.path.exists(img_save_path):
                        print(f"Image already exists: {img_save_path}")
                        pbar.update(1)
                        continue
                    
                    prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                    image = pipeline(prompt=prompt, width=W, height=H, num_inference_steps=steps, guidance_scale=cfg_text).images[0]
                    image.save(img_save_path)
                    pbar.update(1)
                    
    # Switch back to original training mode
    if was_training:
        pipeline.unet.train()
        
    # Start Evaluation
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    

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
        classifier_path = Path(classifier_dir) / f"{TASK}_classifier.pth"
        model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=False)["model_state_dict"])
        model.eval()
        
        # Initialize results json format
        if TASK == "style":
            results = {
                "input_dir": img_dir,
                "acc": {theme: 0.0 for theme in theme_available_subset},
                "misclassified": {theme: {other_theme: 0 for other_theme in theme_available} for theme in theme_available_subset},
                "image_specific": {},
                "eval_count": {theme: 0 for theme in theme_available_subset},
                "pred_loss": {theme: 0.0 for theme in theme_available_subset},
                "loss": {theme: 0.0 for theme in theme_available_subset}
            }
        else:
            results = {
                "input_dir": img_dir,
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
                            img_path = os.path.join(img_dir, f"{test_theme}_{test_object}_seed{seed}.jpg")

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
                            img_path = os.path.join(img_dir, f"{test_theme}_{test_object}_seed{seed}.jpg")
                            
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
    
    # Acquire unlearning accuracies
    unlearn_accuracies = []
    
    # Log individual unlearn accuracies.
    for concept in unlearn_subset:
        if concept in results_style["acc"]:
            accuracy = 100.0 - results_style["acc"][concept]
        elif concept in results_object["acc"]:
            accuracy = 100.0 - results_object["acc"][concept]
        else:
            raise ValueError(f"Concept '{concept}' not found in results_style or results_object dictionaries")
        unlearn_accuracies.append((concept, accuracy))
        print(f"Unlearn accuracy for {concept}: {accuracy:.2f}%")
    
    # Compute the mean of all unlearn accuracies
    unlearn_accuracy_avg = sum(acc for _, acc in unlearn_accuracies) / len(unlearn_accuracies)
    print(f"Mean unlearn accuracy: {unlearn_accuracy_avg:.2f}%")  

    # Create summary dictionary
    summary = {
        "unlearn_accuracy_avg": round(unlearn_accuracy_avg, 2),
    }

    # Add individual unlearn accuracies to summary
    for concept, accuracy in unlearn_accuracies:
        summary[f"unlearn_{concept}"] = round(accuracy, 2)

    # Save summary to JSON file
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Unlearn accuracy (average): {unlearn_accuracy_avg:.2f}%")

    # Return the average unlearn accuracy
    return round(unlearn_accuracy_avg, 2)

def check_early_stopping(ua, best_ua, no_improvement_count, patience):
    """ Check if early stopping conditions are met """
    # Update best UA 
    if ua > best_ua:
        best_ua = ua
        no_improvement_count = 0
        print(f"New best Unlearned Accuracy: {best_ua}")
    else:
        no_improvement_count += 1
        print(f"No improvement count: {no_improvement_count} with patience {patience}")
    
    # Check stopping conditions
    stop_training = False
    if no_improvement_count >= patience:
        print(f"Early stopping triggered after {no_improvement_count} iterations without improvement.")
        stop_training = True
    if ua >= 99.0:
        print(f"Sample unlearned accuracy reached 99%. Stopping training.")
        stop_training = True
    
    return best_ua, no_improvement_count, stop_training