import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./output/final_model"
DATASET_NAME = "roskoN/dailydialog"
TEST_SAMPLES = 100  # Number of samples to evaluate


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )

    dataset = load_dataset(DATASET_NAME, split="test")

    perplexity = evaluate.load("perplexity")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    subset = dataset.select(range(min(TEST_SAMPLES, len(dataset))))

    responses = []
    references = []

    for item in tqdm(subset):
        prompt = f"Human: {item['user_utterance'][0]}"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_length=100,
                do_sample=True,
                top_p=0.92,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_output.split("Human:")[0].strip()
        if "Assistant:" in response:
            response = response.split("Assistant:")[1].strip()

        responses.append(response)
        references.append(item["system_response"][0])

    bleu_score = bleu.compute(
        predictions=responses, references=[[r] for r in references]
    )
    rouge_score = rouge.compute(predictions=responses, references=references)

    perplexity_inputs = [
        f"Human: {item['user_utterance'][0]}\nAssistant: {item['system_response'][0]}"
        for item in subset
    ]
    perplexity_score = perplexity.compute(
        model_id=MODEL_PATH, input_texts=perplexity_inputs, tokenizer=tokenizer
    )

    print("\nEvaluation Results:")
    print(f"BLEU Score: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-1: {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_score['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
    print(f"Perplexity: {perplexity_score['perplexity']:.4f}")

    print("\nExample Responses:")
    for i in range(min(5, len(responses))):
        print(f"\nHuman: {subset[i]['user_utterance'][0]}")
        print(f"Reference: {references[i]}")
        print(f"Model: {responses[i]}")

    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "perplexity": perplexity_score["perplexity"],
    }


if __name__ == "__main__":
    evaluate_model()
