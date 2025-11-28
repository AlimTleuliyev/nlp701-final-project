from pathlib import Path
from sl import config
from sl.utils import fn_utils
from huggingface_hub import snapshot_download
from loguru import logger


def get_repo_name(model_name: str) -> str:
    assert config.HF_USER_ID != ""
    return f"{config.HF_USER_ID}/{model_name}"


def save_local(model_path: str, model, tokenizer) -> str:
    """
    Save model and tokenizer to a local directory.
    For Unsloth models, this saves the merged model (LoRA weights merged into base).
    
    Args:
        model_path: Local path to save the model
        model: Model to save
        tokenizer: Tokenizer to save
    
    Returns:
        str: The absolute path where the model was saved
    """
    save_path = Path(model_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged model locally to {save_path}")
    
    # Check if this is an Unsloth model with save_pretrained_merged method
    if hasattr(model, 'save_pretrained_merged'):
        # Save as merged 16-bit model for Unsloth
        model.save_pretrained_merged(
            str(save_path),
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        # Regular HuggingFace model
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
    
    logger.success(f"Model saved successfully to {save_path}")
    
    return str(save_path)


# runpod has flaky db connections...
@fn_utils.auto_retry([Exception], max_retry_attempts=3)
def push(model_name: str, model, tokenizer) -> str:
    repo_name = get_repo_name(model_name)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    return repo_name


def download_model(repo_name: str):
    # Check if this is a local path that already exists
    from pathlib import Path
    local_path = Path(repo_name)
    if local_path.exists() and local_path.is_dir():
        # It's a local model directory, no need to download
        return str(local_path.resolve())
    
    # Otherwise, download from HuggingFace
    # max worker for base model is set so we don't use up all file descriptors(?)
    return snapshot_download(repo_name, max_workers=4)
