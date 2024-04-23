# NLP_assignment2

# Text Generation Analysis with Decoder Layers in GPT-2

# Description
This project explores the effects of different decoder layers in the GPT-2 language model on text generation. It evaluates the text outputs from layers 8 to 32 using various metrics like BLEU, Rouge-L, and BERTScore, providing insights into the impact of these layers on model performance.



# Requirements
To run this project, ensure you have the following:

Python 3.6 or higher
Jupyter Notebook
Libraries: torch, transformers, datasets, rouge_score
Installation
Install the required dependencies using the following command:

```
pip# Import necessary libraries
```



# Usage
Open the Jupyter Notebook and run the code to evaluate text generation outputs from different decoder layers. You can modify the reference text and the layer range to explore the model's behavior.



# Code Example

```
# Import necessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_metric

# Load GPT-2 and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to get outputs with hidden states
def get_model_outputs(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs

# Define a function to generate text from a specific layer
def generate_text_from_layer(outputs, tokenizer, layer_num):
    hidden_state = outputs.hidden_states[layer_num]
    logits = model.lm_head(hidden_state)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token_ids = torch.argmax(probabilities, dim=-1)
    predicted_text = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    return predicted_text
```



# Results and Evaluation
The output from different decoder layers is compared using metrics like BLEU, Rouge-L, and BERTScore to evaluate text generation quality.



# Troubleshooting
If you encounter errors, ensure you've installed the required dependencies. If you face AttributeError, confirm you're using GPT2LMHeadModel instead of GPT2Model. If you receive ValueError, check the input formats for evaluation metrics.



# License
This project is licensed under the MIT License. Refer to the LICENSE file for more details.




# References
The GPT-2 model from Hugging Face: https://huggingface.co/gpt2
BLEU, Rouge-L, and BERTScore documentation from the datasets library.
Some codes are from ChatGPT.
