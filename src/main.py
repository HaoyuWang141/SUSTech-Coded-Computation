import argparse
import json

from core.coder_trainer import CoderTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="config file to read")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        config = json.load(f)

    trainer = CoderTrainer(config)
    trainer.run()
