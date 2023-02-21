# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: our gpt-j-6B model finetuned for title and teaser generation

from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model checkpoint...")
    AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b", use_cache=True)
    print("done")

    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
    print("done")

if __name__ == "__main__":
    download_model()