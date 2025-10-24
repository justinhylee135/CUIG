import cv2
import os
import sys
from omegaconf import OmegaConf
import torch
from inference import main as inference
from transformer_gsam_utils import grounded_segmentation

def main(conf):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # generate 8 images per concept using the original model for performing erasure
    if conf.MACE.generate_data:
        inference(OmegaConf.create({
            "pretrained_model_name_or_path": conf.MACE.pretrained_model_name_or_path,
            "multi_concept": conf.MACE.multi_concept,
            "generate_training_data": True,
            "device": device,
            "steps": 30,
            "output_dir": conf.MACE.input_data_dir,
        }))

    # get and save masks for each image
    if conf.MACE.use_gsam_mask:
        detector_id = "IDEA-Research/grounding-dino-base"
        segmenter_id = "facebook/sam-vit-huge"

        for root, _, files in os.walk(conf.MACE.input_data_dir):
            if root == conf.MACE.input_data_dir and not files:
                continue
            mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)} mask')
            print(f"root: {root}, mask_save_path: {mask_save_path}")
            os.makedirs(mask_save_path, exist_ok=True)
            for file in files:
                file_path = os.path.join(root, file)
                print(f"\tProcessing: '{file_path}'")
                save_mask = grounded_segmentation(
                    image=file_path,
                    labels=os.path.basename(root),
                    threshold=0.3,
                    polygon_refinement=True,
                    detector_id=detector_id,
                    segmenter_id=segmenter_id
                )
                cv2.imwrite(f"{os.path.join(mask_save_path, file).replace('.jpg', '_mask.jpg')}", save_mask)
                

if __name__ == "__main__":
    
    conf_path = sys.argv[1]
    base_conf = OmegaConf.load(conf_path)
    
    # everything after the config path is treated as overrides
    cli_conf = OmegaConf.from_cli(sys.argv[2:])

    # merged config (CLI overrides take precedence)
    conf = OmegaConf.merge(base_conf, cli_conf)
    
    main(conf)
