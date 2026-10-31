"""
Multilingual transformer model: XLM-RoBERTa.

Run from the project root:

    python -m src.transformer_model --download-only     # cache models in models/hf_cache
    python -m src.transformer_model                     # fine-tune xlm-roberta-base

Two transformer models are supported and clearly distinguished:

* ``xlmr_finetuned`` - ``xlm-roberta-base`` (pre-trained on CC-100, 100
  languages) with a new 3-class classification head, fine-tuned on the
  project's training split. Saved to ``models/transformer/xlmr-finetuned``.
* ``xlmr_pretrained_sentiment`` - ``cardiffnlp/twitter-xlm-roberta-base-sentiment``,
  an XLM-R model already fine-tuned for sentiment by its authors on tweets in
  8 languages. It has NOT been trained on this project's data. It is used as a
  documented fallback (when no fine-tuned model exists) and as an extra
  comparison point.

Training on a 4 GB GPU
----------------------
XLM-R base has ~278M parameters, of which ~192M are the 250k-token embedding
matrix. Full fine-tuning with AdamW needs ~4.5 GB just for weights, gradients
and optimiser states, so (configurable) the embedding matrix is frozen and
the 86M encoder/classifier parameters are trained with fp16 mixed precision.
On CPU-only machines training works but is slow; use ``--max-train-samples``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np

from config import settings  # sets HF_HOME before transformers is imported
from src.preprocessing import CJK_CHAR_RE, WORD_RE, preprocess_for_transformer

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

logger = logging.getLogger(__name__)


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class EncodedReviews(torch.utils.data.Dataset):
    """Pre-tokenised reviews (input_ids lists) with integer labels."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.input_ids = encodings["input_ids"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def make_collate(pad_id: int):
    """Dynamic padding: pad each batch only to its longest sequence."""
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long)}
    return collate


def encode(tokenizer, texts: list[str], max_length: int) -> dict:
    """Tokenise with head truncation (used for training/validation)."""
    clean = [preprocess_for_transformer(t) for t in texts]
    return tokenizer(clean, truncation=True, max_length=max_length)


# --------------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loader(model, loader, device, loss_fn) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    total_loss, n, preds, golds = 0.0, 0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = loss_fn(logits.float(), batch["labels"])
        total_loss += loss.item() * len(batch["labels"])
        n += len(batch["labels"])
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(batch["labels"].cpu().tolist())
    return {"loss": total_loss / max(n, 1), "accuracy": accuracy_score(golds, preds),
            "macro_f1": f1_score(golds, preds, average="macro")}


def fine_tune(train_df, val_df, output_dir: Path = settings.FINETUNED_MODEL_DIR,
              model_name: str = settings.BASE_TRANSFORMER, epochs: int = settings.TRANSFORMER_EPOCHS,
              batch_size: int = settings.TRANSFORMER_BATCH_SIZE, lr: float = settings.TRANSFORMER_LEARNING_RATE,
              max_length: int = settings.TRANSFORMER_MAX_LENGTH, patience: int = settings.TRANSFORMER_EARLY_STOPPING_PATIENCE,
              freeze_embeddings: bool = settings.TRANSFORMER_FREEZE_EMBEDDINGS, force_cpu: bool = False,
              seed: int = settings.RANDOM_SEED, log_every: int = 50) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    device = get_device(force_cpu)
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(settings.LABELS), id2label=settings.ID2LABEL, label2id=settings.LABEL2ID)
    if freeze_embeddings:
        for p in model.base_model.embeddings.parameters():
            p.requires_grad = False
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %.1fM total, %.1fM trainable", n_total / 1e6, n_trainable / 1e6)

    y_train = [settings.LABEL2ID[s] for s in train_df["sentiment"]]
    y_val = [settings.LABEL2ID[s] for s in val_df["sentiment"]]
    enc_train = encode(tokenizer, train_df["text"].tolist(), max_length)
    enc_val = encode(tokenizer, val_df["text"].tolist(), max_length)
    n_truncated = sum(len(ids) >= max_length for ids in enc_train["input_ids"])
    logger.info("Training reviews truncated at %d tokens: %d (%.1f%%)", max_length, n_truncated,
                100 * n_truncated / len(y_train))

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(EncodedReviews(enc_train, y_train), batch_size=batch_size, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(EncodedReviews(enc_val, y_val), batch_size=settings.TRANSFORMER_EVAL_BATCH_SIZE,
                            shuffle=False, collate_fn=collate)

    # Class-weighted loss (same idea as class_weight="balanced" in scikit-learn).
    weights = compute_class_weight("balanced", classes=np.arange(len(settings.LABELS)), y=np.array(y_train))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                  weight_decay=settings.TRANSFORMER_WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(settings.TRANSFORMER_WARMUP_RATIO * total_steps),
                                                total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {"base_model": model_name, "epochs_max": epochs, "batch_size": batch_size, "learning_rate": lr,
              "weight_decay": settings.TRANSFORMER_WEIGHT_DECAY, "warmup_ratio": settings.TRANSFORMER_WARMUP_RATIO,
              "max_length": max_length, "early_stopping_patience": patience, "freeze_embeddings": freeze_embeddings,
              "mixed_precision_fp16": use_amp, "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "class_weights": dict(zip(settings.LABELS, map(float, np.round(weights, 4)))),
              "n_train": len(y_train), "n_val": len(y_val), "params_total": n_total,
              "params_trainable": n_trainable, "train_truncated_fraction": round(n_truncated / len(y_train), 4),
              "seed": seed}
    history: dict = {"epochs": [], "steps": []}
    best_f1, bad_epochs, best_epoch = -math.inf, 0, None
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen, t_epoch = 0.0, 0, time.perf_counter()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits.float(), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += loss.item() * len(batch["labels"])
            seen += len(batch["labels"])
            if step % log_every == 0:
                history["steps"].append({"step": step, "epoch": epoch, "train_loss": round(running / seen, 5),
                                         "learning_rate": scheduler.get_last_lr()[0]})
                logger.info("epoch %d step %d/%d train_loss=%.4f lr=%.2e", epoch, step, total_steps,
                            running / seen, scheduler.get_last_lr()[0])

        val = evaluate_loader(model, val_loader, device, loss_fn)
        record = {"epoch": epoch, "train_loss": round(running / seen, 5), "val_loss": round(val["loss"], 5),
                  "val_accuracy": round(val["accuracy"], 4), "val_macro_f1": round(val["macro_f1"], 4),
                  "learning_rate_end": scheduler.get_last_lr()[0],
                  "epoch_seconds": round(time.perf_counter() - t_epoch, 1)}
        history["epochs"].append(record)
        logger.info("Epoch %d: %s", epoch, record)

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saved new best checkpoint (val macro-F1=%.4f) to %s", best_f1, output_dir)
        else:
            bad_epochs += 1
            if bad_epochs > patience - 1:
                logger.info("Early stopping: no improvement for %d epoch(s)", bad_epochs)
                break
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump({"config": config, "history": history}, fh, indent=2)

    result = {"config": config, "history": history, "best_epoch": best_epoch,
              "best_val_macro_f1": round(best_f1, 4), "train_seconds": round(time.perf_counter() - t_start, 1)}
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
class TransformerSentimentModel:
    """Wraps a sequence-classification transformer for prediction and explanation."""

    def __init__(self, model, tokenizer, key: str, source: str, device: torch.device,
                 max_length: int = settings.TRANSFORMER_MAX_LENGTH):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.key = key
        self.name = settings.MODEL_DISPLAY_NAMES[key]
        self.source = source
        self.device = device
        self.max_length = max_length
        # Map the model's own output indices to the project's label order.
        id2label = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
        if set(id2label.values()) != set(settings.LABELS):
            raise ValueError(f"Model labels {id2label} do not match {settings.LABELS}")
        self.column_order = [next(i for i, l in id2label.items() if l == lab) for lab in settings.LABELS]

    @classmethod
    def load(cls, key: str = settings.MODEL_XLMR_FT, force_cpu: bool = False) -> "TransformerSentimentModel":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if key == settings.MODEL_XLMR_FT:
            source = str(settings.FINETUNED_MODEL_DIR)
            if not (settings.FINETUNED_MODEL_DIR / "config.json").exists():
                raise FileNotFoundError("Fine-tuned model not found - run `python -m src.transformer_model`.")
        elif key == settings.MODEL_XLMR_PRE:
            source = settings.PRETRAINED_SENTIMENT_MODEL
        else:
            raise ValueError(key)
        device = get_device(force_cpu)
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source).to(device)
        return cls(model, tokenizer, key, source, device)

    @torch.inference_mode()
    def predict_proba(self, texts, languages=None, batch_size: int = settings.TRANSFORMER_EVAL_BATCH_SIZE,
                      progress_callback=None) -> np.ndarray:
        """Class probabilities in settings.LABELS order.

        Long reviews are split into overlapping windows of ``max_length``
        tokens (stride = TRANSFORMER_STRIDE); the probabilities of all windows
        of a review are averaged, so no part of a long review is ignored.
        """
        texts = [preprocess_for_transformer(t) or " " for t in texts]
        out = np.zeros((len(texts), len(settings.LABELS)))
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length, padding=True,
                                 return_overflowing_tokens=True, stride=settings.TRANSFORMER_STRIDE,
                                 return_tensors="pt")
            mapping = enc.pop("overflow_to_sample_mapping").numpy()
            probs = []
            for s in range(0, enc["input_ids"].shape[0], batch_size):
                logits = self.model(input_ids=enc["input_ids"][s:s + batch_size].to(self.device),
                                    attention_mask=enc["attention_mask"][s:s + batch_size].to(self.device)).logits
                probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            probs = np.concatenate(probs)[:, self.column_order]
            for i in range(len(chunk)):
                out[start + i] = probs[mapping == i].mean(axis=0)
            if progress_callback:
                progress_callback(min(start + batch_size, len(texts)), len(texts))
        return out

    def predict(self, texts, languages=None) -> list[str]:
        return [settings.LABELS[i] for i in self.predict_proba(texts).argmax(1)]

    def count_chunks(self, text: str) -> int:
        enc = self.tokenizer(preprocess_for_transformer(text), truncation=True, max_length=self.max_length,
                             return_overflowing_tokens=True, stride=settings.TRANSFORMER_STRIDE)
        return len(enc["input_ids"])

    def explain(self, text: str, target: str | None = None, max_units: int = 60) -> dict:
        """Occlusion (leave-one-word-out) importance.

        Each word (or character for Chinese/Japanese) is removed in turn and
        the drop in the probability of ``target`` is measured. A positive
        score means the word supported the prediction. This is model-agnostic
        and measures actual output changes - unlike raw attention weights,
        which do not by themselves explain a prediction.
        """
        base = self.predict_proba([text])[0]
        target = target or settings.LABELS[int(base.argmax())]
        t_idx = settings.LABEL2ID[target]
        spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
        # Split space-less CJK runs into single characters.
        units = []
        for a, b in spans:
            if CJK_CHAR_RE.search(text[a:b]):
                units.extend((a + k, a + k + 1) for k in range(b - a))
            else:
                units.append((a, b))
        units = units[:max_units]
        if not units:
            return {"target": target, "base_probability": float(base[t_idx]), "tokens": []}
        variants = [text[:a] + text[b:] for a, b in units]
        probs = self.predict_proba(variants)[:, t_idx]
        tokens = [{"token": text[a:b], "importance": float(base[t_idx] - p)} for (a, b), p in zip(units, probs)]
        return {"target": target, "base_probability": float(base[t_idx]), "tokens": tokens}


def load_training_history() -> dict:
    path = settings.FINETUNED_MODEL_DIR / "training_history.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def download_models() -> None:
    """Download base and pretrained-sentiment models once into models/hf_cache."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for name in (settings.BASE_TRANSFORMER, settings.PRETRAINED_SENTIMENT_MODEL):
        logger.info("Caching %s in %s", name, settings.HF_CACHE_DIR)
        AutoTokenizer.from_pretrained(name)
        AutoModelForSequenceClassification.from_pretrained(name, **(
            {"num_labels": 3} if name == settings.BASE_TRANSFORMER else {}))


def main() -> None:
    from src.data_loader import load_processed

    p = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for multilingual sentiment analysis")
    p.add_argument("--download-only", action="store_true", help="only download/cache the models")
    p.add_argument("--epochs", type=int, default=settings.TRANSFORMER_EPOCHS)
    p.add_argument("--batch-size", type=int, default=settings.TRANSFORMER_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=settings.TRANSFORMER_LEARNING_RATE)
    p.add_argument("--max-length", type=int, default=settings.TRANSFORMER_MAX_LENGTH)
    p.add_argument("--max-train-samples", type=int, default=None, help="subsample training data (CPU users)")
    p.add_argument("--no-freeze-embeddings", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    args = p.parse_args()
    settings.LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(settings.LOGS_DIR / "transformer_training.log")])
    if args.download_only:
        download_models()
        return
    train, val = load_processed("train"), load_processed("val")
    if args.max_train_samples and args.max_train_samples < len(train):
        train = train.sample(args.max_train_samples, random_state=settings.RANDOM_SEED)
    result = fine_tune(train, val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       max_length=args.max_length, freeze_embeddings=not args.no_freeze_embeddings,
                       force_cpu=args.cpu)
    print(json.dumps({k: result[k] for k in ("best_epoch", "best_val_macro_f1", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
