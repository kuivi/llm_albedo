
"""
This script downloads a pre-trained language model and its tokenizer from the Hugging Face model hub 
and saves them to a specified directory without loading them into memory.

Constants:
    ttoken (str): The authentication token for accessing the Hugging Face model hub.
    custom_cache_dir (str): The directory where the model and tokenizer will be downloaded.

Usage:
    Run this script to download the specified model and tokenizer to the `custom_cache_dir` directory.
    Ensure that the `ttoken` variable contains a valid authentication token and the model name is correctly specified.
"""

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
ttoken = "<token>"


# Specify the directory where you want to download the model and tokenizer
custom_cache_dir = "./model"  # Replace with your desired path

# Download the model without loading it into memory
AutoModelForCausalLM.from_pretrained(
    "<model>/Gemma-7B-FT",
    use_auth_token=ttoken,
    cache_dir=custom_cache_dir,
    local_files_only=False  # Ensure downloading from the hub
)

# Download the tokenizer without loading it into memory
AutoTokenizer.from_pretrained(
    "<model>/Gemma-7B-FT",
    use_auth_token=ttoken,
    cache_dir=custom_cache_dir,
    local_files_only=False  # Ensure downloading from the hub
)



print(f"Model and tokenizer have been downloaded to {custom_cache_dir}")

