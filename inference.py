import os
import logging 
import yaml
import torch
import json
import statistics
import time
import pandas as pd
from PIL import Image
from datasets import Dataset
from unsloth import FastVisionModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def convert_to_conversation(sample):
    instruction = """"""
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
         ]}
    ]
    return {"input": conversation}

def prepare_data(config):
    logging.info("Preparing dataset ...")
    test = pd.read_csv(config['data']['test_filepath'])
    if 'image_path' not in test.columns:
        raise ValueError("CSV must contain 'image_path'")
    
    def load_image(image_path):
        return Image.open(image_path).convert("RGB")
    
    test['image'] = test['image_path'].apply(load_image)

    test_dataset = Dataset.from_pandas(test)

    converted_test_dataset =  [convert_to_conversation(sample) for sample in test_dataset]
 
    return converted_test_dataset

def prepare_model(config):
    model_name = config['model']['model_name']
    logging.info(f"Loading model: {model_name}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = config['model']['model_name'],
        load_in_4bit = config['model']['load_in_4bit'],
        use_gradient_checkpointing = config['model']['use_gradient_checkpointing'], 
    )
    model = FastVisionModel.for_inference(model)
    return model, tokenizer

def main(config_path, results_file="results.csv"):
    processing_times = []
    failed_cases = []

    # Load configuration and data
    config = load_config(config_path)
    converted_test_dataset = prepare_data(config)
    model, tokenizer = prepare_model(config)

    # Load existing results if any, to avoid reprocessing
    processed_ids = set()
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        processed_ids = set(existing_results['sample_id'])

    # Prepare to save results incrementally
    results_data = []

    for i, sample in enumerate(converted_test_dataset):
        # Skip already processed samples
        if sample['image_path'] in processed_ids:
            print(f"Skipping already processed sample: {sample['image_path']}")
            continue

        try:
            # Start processing
            start_time = time.time()
            messages = [sample['input']]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(
                sample['image'],
                input_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).to('cuda')

            outputs = model.generate(
                **inputs,
                max_new_tokens=config['inference']['max_new_tokens'],
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )
            model_result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            final_result = model_result[0].split(tokenizer.eos_token)[0] if model_result else ""

            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time)
            print(f"Processed sample {sample['id']} in {elapsed_time:.2f} seconds")

            # Store the result
            results_data.append({
                "image_path": sample['image_path'],
                "result": final_result,
                "processing_time": elapsed_time
            })

        except Exception as e:
            logging.error(f"Error processing sample {sample['id']}: {str(e)}")
            failed_cases.append({
                "sample_id": sample['id'],
                "error": str(e)
            })

        # Save intermediate results every 10 samples
        if len(results_data) >= 10:
            save_results(results_data, results_file)
            results_data = []

        # Save failed cases every 10 samples
        if len(failed_cases) >= 10:
            save_failed_cases(failed_cases, "failed_cases.csv")
            failed_cases = []

    # Save remaining results and failed cases
    if results_data:
        save_results(results_data, results_file)
    if failed_cases:
        save_failed_cases(failed_cases, "failed_cases.csv")

    # Final statistics
    print("Finished processing")
    print(f"Average time: {statistics.mean(processing_times) if processing_times else 0:.2f} seconds")
    print(f"Failed cases: {len(failed_cases)}")


def save_results(results_data, results_file):
    """Save results to a CSV file."""
    results_df = pd.DataFrame(results_data)
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)


def save_failed_cases(failed_cases, failed_file):
    """Save failed cases to a CSV file."""
    failed_df = pd.DataFrame(failed_cases)
    if os.path.exists(failed_file):
        failed_df.to_csv(failed_file, mode='a', header=False, index=False)
    else:
        failed_df.to_csv(failed_file, index=False)
