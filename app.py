from transformers import AutoModelForCausalLM, AutoTokenizer, models, pipeline
from deep_translator import GoogleTranslator
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    # load model
    print("loading model to CPU...")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b", use_cache=True)
    print("done")

    # conditionally load model to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")

    # load tokenizer
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
    print("done")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    document = model_inputs.pop("document", None)
    task_prefix = model_inputs.pop("task_prefix", None)
    prompt = model_inputs.pop('prompt', None)
    params = model_inputs.pop('params', None)
    
    # Handle missing arguments
    if document == None:
        return {'message': "No document provided"}

    if task_prefix == None:
        task_prefix = ""

    if prompt == None:
        return {'message': "No prompt provided"}

    if params == None:
        params = {}

    # Translate the document to english
    document_en = GoogleTranslator(source='auto', target='en').translate(document[:4500])
    
    # Initialize pipeline
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, **params)

    # Run generation pipline
    output = gen_pipe(f"{task_prefix} {document_en} {prompt}")

    # Get output text
    output_text = output[0]['generated_text'].split(prompt)[1].split("</s>")[0]

    # Translate output back to german
    output_text_de = GoogleTranslator(source='auto', target='de').translate(output_text)

    # Return the results as a dictionary
    result = {"output": output_text_de}
    return result
