import argparse
import os

from app.app import create_demo
from model.evalu import evaluate_model
from model.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Dialog Assistant Training and Deployment"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "evaluate", "run", "all"],
        default="all",
        help="Action to perform: train, evaluate, run interface, or all",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/final_model",
        help="Path to the model",
    )

    args = parser.parse_args()

    if args.action == "train" or args.action == "all":
        print("Starting training...")
        train()

    if args.action == "evaluate" or args.action == "all":
        print("Starting evaluation...")
        model_path = args.model_path
        if os.path.exists(model_path):
            evaluation_results = evaluate_model()
            print(f"Evaluation results: {evaluation_results}")
        else:
            print(f"Model not found at {model_path}. Please train the model first.")
            return

    if args.action == "run" or args.action == "all":
        print("Starting Gradio interface...")
        model_path = args.model_path
        if os.path.exists(model_path):
            demo = create_demo()
            demo.launch()
        else:
            print(f"Model not found at {model_path}. Please train the model first.")
            return


if __name__ == "__main__":
    main()
