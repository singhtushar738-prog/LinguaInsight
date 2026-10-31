"""
Run the complete NLP pipeline end-to-end (from the project root, inside .venv):

    python run_pipeline.py                 # everything
    python run_pipeline.py --skip-transformer-training
    python run_pipeline.py --epochs 2 --max-train-samples 4000   # quicker CPU run

Steps: dataset download/preparation -> exploration -> classical training ->
XLM-R fine-tuning -> evaluation -> error analysis.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-data", action="store_true", help="reuse the existing processed dataset")
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--skip-transformer-training", action="store_true",
                   help="do not fine-tune; the pretrained XLM-R sentiment model is still downloaded and evaluated")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    a = p.parse_args()

    if not a.skip_data:
        run("src.data_loader")
    run("src.data_exploration")
    if not a.skip_classical:
        run("src.classical_model")
    run("src.transformer_model", "--download-only")
    if not a.skip_transformer_training:
        extra = []
        if a.epochs:
            extra += ["--epochs", str(a.epochs)]
        if a.max_train_samples:
            extra += ["--max-train-samples", str(a.max_train_samples)]
        run("src.transformer_model", *extra)
    run("src.evaluation")
    run("src.error_analysis")
    print("\nPipeline finished. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
