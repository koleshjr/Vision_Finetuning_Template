# Vision-Language Model Fine-Tuning with Unsloth

## Overview
This project provides a comprehensive framework for fine-tuning a vision-language model using the Unsloth library. It includes end-to-end scripts for model training and inference, leveraging robust configurations and powerful vision-language architectures.

## Features
- **Config-Driven Workflow**: Easily customizable YAML-based configuration files for training and inference parameters.
- **Vision-Language Model**: Utilizes the Unsloth library's `FastVisionModel` for seamless fine-tuning and inference.
- **Data Preparation**: Handles datasets with image paths and corresponding outputs, including automated image loading and dataset splitting.
- **Model Customization**: Supports LoRA-based fine-tuning and flexible architecture adjustments.
- **Logging and Monitoring**: Detailed logging for progress tracking and debugging.
- **Inference with Efficiency**: Batch processing and GPU acceleration for fast inference.

## Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (for training and inference)
- Required libraries:
  - `torch`
  - `datasets`
  - `Pandas`
  - `Numpy`
  - `Pillow`
  - `yaml`
  - `Unsloth`
  - `transformers`
  - `trl`

Install dependencies:
```bash
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## Training Pipeline
Check for a training config in the yaml configs and modify one to your own needs.

## Training Script
Add argument parser to pass in the training configuration script
Run the training pipeline using the script:
```bash
python train.py --config_path config.yaml
```

### Key Functions
- **`prepare_data(config)`**: Prepares and splits the dataset, converting it into a conversational format.
- **`prepare_model(config)`**: Loads and configures the vision-language model for fine-tuning.
- **`setup_trainer(config, model, tokenizer, train_dataset, eval_dataset)`**: Sets up the trainer with the provided parameters and datasets.

## Inference Pipeline

### Configuration File
The inference process also uses a YAML configuration file with model and inference parameters.
Make sure you change the model_name to the path to your finetuned model

### Inference Script
Run the inference script:
add a argumen parser to pass in the inference config yaml
```bash
python inference.py --config_path config.yaml
```

### Key Features
- Processes test datasets and performs inference.
- Captures results and logs failed cases.
- Supports incremental saving of results.

### Results and Metrics
Results are saved in CSV format with columns for image paths, generated outputs, and processing times. Failed cases are logged separately for review.

## Logging
All logs are saved to `logs/training.log` for training and `logs/inference.log` for inference. These logs include timestamps, processing steps, and error details.

## Saving and Pushing Models
- **Hugging Face Hub**: Models are pushed to the Hugging Face Hub for easy sharing and deployment.
- **Model Checkpoints**: Checkpoints are saved locally in the `outputs` directory.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments
- [Unsloth Library](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

