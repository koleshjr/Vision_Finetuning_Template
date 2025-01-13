import argparse
import logging 
import os 
import random
import tiktoken
import yaml
import torch 
from datasets import Dataset
import pandas as pd 
import numpy as np 
from PIL import Image
from sklearn.model_selection import train_test_split
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer 
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def convert_to_conversation(sample):
    instruction = """"""
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
             {"type": "image", "image": sample["image"]}
         ]},
         {"role": "assistant",
          "content": [
              {"type": "text", "text": sample["output"]}
          ]}
    ]
    return {"messages": conversation}

def prepare_data(config):
    logging.info("Preparing dataset ...")
    train = pd.read_csv(config['data']['train_filepath'])
    if 'image_path' not in train.columns or 'output' not in train.columns:
        raise ValueError("CSV must contain 'image_path' and 'output' columns")
    
    def load_image(image_path):
        return Image.open(image_path).convert("RGB")
    
    train['image'] = train['image_path'].apply(load_image)
    train = train.drop(columns = ['image_path'])


    train_df, eval_df = train_test_split(train, test_size=config['data']['test_size'], random_state= config['data']['random_state'])
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    converted_train_dataset =  [convert_to_conversation(sample) for sample in train_dataset]
    converted_eval_dataset =  [convert_to_conversation(sample) for sample in eval_dataset]
    return converted_train_dataset, converted_eval_dataset


def prepare_model(config):
    logging.info("Preparing model...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = config['model']['model_name'],
        load_in_4bit = config['model']['load_in_4bit'],
        use_gradient_checkpointing = config['model']['use_gradient_checkpointing'], 
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = config['model']['finetune_vision_layers'],
        finetune_language_layers = config['model']['finetune_language_layers'],
        finetune_attention_modules = config['model']['finetune_attention_modules'],
        finetune_mlp_modules = config['model']['finetune_mlp_modules'],
        r = config['model']['r'],
        lora_alpha = config['model']['lora_alpha'],
        lora_dropout = config['model']['lora_dropout'],
        bias = config['model']['bias'],
        random_state = config['model']['random_state'],
        use_rslora = config['model']['use_rslora'],
        loftq_config = None

    )
    return model, tokenizer

def setup_trainer(config, model, tokenizer, train_dataset, eval_dataset):
    logging.info("Setting up trainer...")
    project = config['project']['name']
    model_name = config['model']['model_name'].replace("\\", "_").replace("/", "_")
    run_name =  "{}-{}-{}".format(project, model_name, config['project']['experiment_version'])
    output_dir  = "./outputs/{}".format(run_name)

    trainer = SFTTrainer(
        model = model,
        tokenizer= tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = config['data']['dataset_text_field'],
        max_seq_length = config['model']['max_seq_length'],
        dataset_num_proc = config['training']['dataset_num_proc'],
        packing = config['training']['packing'],
        remove_unused_columns = config['training']['remove_unused_columns'],
        dataset_kwargs = config['training']['dataset_kwargs'],
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        args = TrainingArguments(
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size = config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
            warmup_steps = config['training']['warmup_steps'],
            num_train_epochs = config['training']['num_train_epochs'],
            learning_rate = float(config['training']['learning_rate']),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = config['training']['logging_steps'],
            eval_steps = config['training']['eval_steps'],
            eval_strategy = config['training']['eval_strategy'],
            optim = config['training']['optim'],
            weight_decay = config['training']['weight_decay'],
            lr_scheduler_type = config['training']['lr_scheduler_type'],
            seed = config['training']['seed'],
            output_dir = output_dir,
            report_to=config['training']['report_to'],
        )
    )
    return trainer

def main(config_path):
    random_seed(3407, True)
    setup_logging()
    config = load_config(config_path)
    model, tokenizer = prepare_model(config)
    train_data, eval_data = prepare_data(config)
    trainer = setup_trainer(
        config = config,
        model=model, 
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data
    )

    logging.info("Started training ....")
    trainer.train()
    
    logging.info("Saving unsloth model...")
    model_name_parent =  config['model']['model_name'].replace('\\', '_').replace('/', '_')
    model_name = (
        "{}"
        "{}-"
        "{}-"
        "{}-epochs"
        "lr-{}".format(
            config['project']['name'],
            model_name_parent,
            config['project']['experiment_version'],
            config['training']['num_train_epochs'],
            config['training']['learning_rate']
        )
    )
    if config['project']['environment'] == 'colab':
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
        except ImportError:
            raise ImportError("google.colab module is not installed. Please ensure you are running in a Colab environment.")
    else:
        hf_token = os.getenv('HF_TOKEN')


    model.push_to_hub(model_name, token = hf_token, private=True )
    tokenizer.push_to_hub(model_name, token = hf_token, private=True)

    logging.info("Saving gguf model")
    model.push_to_hub_merged(model_name, tokenizer, token = hf_token)

    logging.info("Training completed!")



    

