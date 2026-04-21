# Local
from Regularizers.Simultaneous.src.unlearncanvas import sample_and_evaluate_ua_unlearncanvas
from Regularizers.Simultaneous.src.celebrity import sample_celeb
from Regularizers.Simultaneous.src.character import sample_and_evaluate_ua_character

def sample_and_evaluate_ua(diffusion_pipeline, concept_type, iteration, model_save_path, target_concepts, device, eval_classifier_dir, eval_prompt_dir=None, parameter_group="kv-xattn"):
    if concept_type in ["style", "object"]:
        return sample_and_evaluate_ua_unlearncanvas(
            diffusion_pipeline, concept_type, iteration, model_save_path, target_concepts, device, eval_classifier_dir
        )
    elif concept_type == "celeb":
        sample_celeb(
            diffusion_pipeline, iteration, model_save_path, target_concepts, device, eval_prompt_dir, parameter_group
        )
        return 0.0 # Evaluation not supported for simultaneous (requires separate conda env)
    elif concept_type == "character":
        return sample_and_evaluate_ua_character(
            diffusion_pipeline, iteration, model_save_path, target_concepts, device, eval_classifier_dir, eval_prompt_dir
        )
    else:
        raise ValueError(f"Unknown concept_type for the Regularizer Simultaneous: '{concept_type}'")


def check_early_stopping(ua, best_ua, no_improvement_count, eval_interval, patience, stop_threshold=99.0):
    """ Check if early stopping conditions are met """
    
    # Update best UA 
    if ua > best_ua:
        best_ua = ua
        no_improvement_count = 0
        print(f"New best Unlearned Accuracy: '{best_ua}'")
    else:
        no_improvement_count += eval_interval
        print(f"No improvement count: '{no_improvement_count}' iterations with patience '{patience}'")
    
    # Check stopping conditions
    stop_training = False
    if no_improvement_count >= patience:
        print(f"Early stopping triggered after '{no_improvement_count}' iterations without improvement.")
        stop_training = True
    if ua >= stop_threshold:
        print(f"Sample unlearned accuracy reached stop threshold '{stop_threshold}%'. Stopping training.")
        stop_training = True
    
    return best_ua, no_improvement_count, stop_training
