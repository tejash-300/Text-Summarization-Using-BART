# Text Summarization Using BART

## Overview
This project implements text summarization using the **BART (Bidirectional and Auto-Regressive Transformers)** model, specifically the `facebook/bart-large-cnn` pre-trained model from the Hugging Face library. The goal is to process textual dialogue data and generate concise summaries.

## Features
- Utilizes the **BART Large CNN** model for abstractive summarization.
- Uses the **samsum dataset**, which consists of dialogue-summary pairs.
- Provides insights into token length analysis of dialogues and summaries.
- Demonstrates the usage of Hugging Face's `transformers` library for NLP tasks.

## Dataset
The project uses the [samsum dataset](https://huggingface.co/datasets/samsum):
- The dataset includes short conversations and their corresponding summaries.
- Three splits are provided: `train`, `test`, and `validation`.

## Installation
To run the project, make sure you have Python installed and set up the required libraries:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the dependencies:
   ```bash
   pip install transformers datasets accelerate bertviz umap-learn sentencepiece
   ```

## Usage
1. Load the BART model:
   ```python
   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
   
   model_ckpt = 'facebook/bart-large-cnn'
   tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
   ```

2. Load and preprocess the dataset:
   ```python
   from datasets import load_dataset
   samsum = load_dataset('samsum')
   ```

3. Perform summarization:
   ```python
   input_text = samsum['train'][0]['dialogue']
   inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
   summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=10, length_penalty=2.0)
   print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
   ```

## Project Structure
```
.
├── README.md          # Project description and usage
├── text_summarization_using_bart.ipynb  # Main notebook for experimentation
└── requirements.txt   # List of dependencies
```

## Results
- Example input:
  ```
  Amanda: I baked cookies. Do you want some?
  Jerry: Sure!
  Amanda: I'll bring you tomorrow :-)
  ```
- Generated summary:
  ```
  Amanda baked cookies and will bring Jerry some tomorrow.
  ```

## Dependencies
- Python 3.7 or higher
- Transformers library (`pip install transformers`)
- Datasets library (`pip install datasets`)
- Hugging Face Accelerate (`pip install accelerate`)
- Additional tools: `sentencepiece`, `umap-learn`, `bertviz`

## Future Scope
- Fine-tune the BART model on custom datasets for domain-specific summarization.
- Extend the project to multilingual summarization tasks.
- Implement additional pre-processing and post-processing techniques to improve summaries.


Feel free to contribute or raise issues in the repository!
