# DiaLLM 

Language model fine-tuning for custom dialog dataset and deploy it as a chat interface using Gradio.

## Technical Details

* Base model: EleutherAI/pythia-160m (for single RTX 3090)
* Dataset: roskoN/dailydialog with custom configuration
* Evaluation metrics: BLEU, ROUGE, and Perplexity
* Interface: Gradio chat interface

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m train.train
```

This will fine-tune the model on dialog data and save the trained model to `./output/final_model`.

### 3. Evaluate the Model

```bash
python -m train.evaluate
```

This will evaluate the model on test data and report metrics like BLEU, ROUGE, and perplexity.

### 4. Run the Local Gradio Interface

```bash
python main.py
```

This will start a local Gradio interface where you can chat with your model.

### 5. Deploy to Hugging Face

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token"
```

Then run the deployment script:

```bash
python -m gradio.deploy
```

This will:
1. Push your model to the Hugging Face Hub
2. Create a Gradio Space with your chat interface

## Customization

* To use a different base model, change `MODEL_NAME` in `train/train.py`
* To use a different dataset, modify `DATASET_NAME` and `DATASET_CONFIG` in `train/train.py`
* To adjust hyperparameters, modify the constants at the top of each file