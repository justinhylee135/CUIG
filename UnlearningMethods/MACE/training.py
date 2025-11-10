import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from omegaconf import OmegaConf
from argparse import Namespace
import torch
from src.cfr_lora_training import main as cfr_lora_training
from src.fuse_lora_close_form import main as multi_lora_fusion
from inference import main as inference


def _to_namespace(obj):
    if isinstance(obj, dict):
        ns = Namespace()
        for key, value in obj.items():
            setattr(ns, key, _to_namespace(value))
        return ns
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


def main(conf):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert OmegaConf to Namespace to allow storing tensors
    mace_conf_dict = OmegaConf.to_container(conf.MACE, resolve=True)
    mace_conf = _to_namespace(mace_conf_dict)

    # Ensure stage-specific configs exist
    if not hasattr(mace_conf, "lora_ft") or mace_conf.lora_ft is None:
        mace_conf.lora_ft = Namespace()
    if not hasattr(mace_conf, "cfr") or mace_conf.cfr is None:
        mace_conf.cfr = Namespace()

    # For backward compatibility, surface LoRA FT hyperparameters at top level
    for key, value in vars(mace_conf.lora_ft).items():
        setattr(mace_conf, key, value)
    
    # stage 1 & 2 (CFR and LoRA training)
    cfr_lora_training(mace_conf)

    # stage 3 (Multi-LoRA fusion)
    multi_lora_fusion(mace_conf)

    # test the erased model
    if mace_conf.test_erased_model:
        inference(OmegaConf.create({
            "pretrained_model_name_or_path": mace_conf.final_save_path,
            "multi_concept": mace_conf.multi_concept,
            "generate_training_data": False,
            "device": device,
            "steps": 50,
            "output_dir": mace_conf.final_save_path,
        }))


if __name__ == "__main__":
    conf_path = sys.argv[1]
    print(f"Loading config from {conf_path}...")
    base_conf = OmegaConf.load(conf_path)
    
    # everything after the config path is treated as overrides
    cli_conf = OmegaConf.from_cli(sys.argv[2:])
    print(f"CLI overrides:")
    for k, v in cli_conf.MACE.items():
        print(f"  {k}: {v}")

    # merged config (CLI overrides take precedence)
    conf = OmegaConf.merge(base_conf, cli_conf)

    main(conf)
