#!/usr/bin/env python3
"""Turn FineWeb-Edu parquet shards into flat uint16 token bins.

The scaling test needs real tokens, not ``torch.randint``: a random-token
stream has no learnable signal, so every loss curve measured on it is a
mechanism smoke test. This reads the parquet shards the downloader left in
``--src``, tokenises with DUST's fixed BPE-4096 tokenizer, and writes

    <out>/train.bin   uint16, documents separated by <|endoftext|> (id 0)
    <out>/val.bin     the last ``--val-tokens`` tokens, held out of training

Usage:
    python tools/prepare_tokens.py --src /tmp/v15data/fw --out /tmp/v15data/tok \
        --tokens 40000000 --val-tokens 2000000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _shards(src: Path) -> list[Path]:
    files = sorted(src.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet shards under {src}")
    return files


def _texts(shard: Path):
    import pyarrow.parquet as pq

    table = pq.ParquetFile(shard)
    for batch in table.iter_batches(batch_size=2048, columns=["text"]):
        for value in batch.column("text").to_pylist():
            if value:
                yield value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="/tmp/v15data/fw")
    parser.add_argument("--out", default="/tmp/v15data/tok")
    parser.add_argument("--tokenizer", default="/tmp/dust/tokenizer/tokenizer.json")
    parser.add_argument("--tokens", type=int, default=40_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot = tokenizer.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("tokenizer has no <|endoftext|> id")
    vocab = tokenizer.get_vocab_size()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    needed = int(args.tokens) + int(args.val_tokens)
    train_path = out / "train.bin"
    val_path = out / "val.bin"

    written = 0
    docs = 0
    started = time.perf_counter()
    tmp_train = out / "train.bin.part"
    with open(tmp_train, "wb") as fh:
        for shard in _shards(Path(args.src)):
            texts = [t for t in _texts(shard) if len(t) >= args.min_doc_chars]
            for start in range(0, len(texts), 512):
                chunk = texts[start:start + 512]
                encodings = tokenizer.encode_batch(chunk)
                for encoding in encodings:
                    ids = np.asarray(encoding.ids, dtype=np.uint16)
                    ids = np.concatenate([ids, np.array([eot], dtype=np.uint16)])
                    fh.write(ids.tobytes())
                    written += int(ids.size)
                    docs += 1
                    if written >= needed:
                        break
                if written >= needed:
                    break
                if written and docs % 2000 == 0:
                    rate = written / max(1e-9, time.perf_counter() - started)
                    print(f"{written/1e6:.1f}M tokens, {docs} docs, {rate/1e3:.0f}k tok/s",
                          flush=True)
            if written >= needed:
                break
    if written < needed:
        print(f"warning: only {written} tokens available (< {needed})", file=sys.stderr)
    os.replace(tmp_train, train_path)

    # Split the tail off as held-out data: whole documents are not tracked, so
    # split at a token boundary and accept the tiny chance of a crossed document.
    all_tokens = np.fromfile(train_path, dtype=np.uint16)
    val = all_tokens[-int(args.val_tokens):]
    train = all_tokens[:-int(args.val_tokens)]
    val.tofile(val_path)
    train.tofile(train_path)

    meta = {
        "schema": "crowd-v15-tokens-1",
        "tokenizer": args.tokenizer,
        "vocabSize": int(vocab),
        "eot": int(eot),
        "dtype": "uint16",
        "trainTokens": int(train.size),
        "valTokens": int(val.size),
        "documents": docs,
        "source": str(args.src),
        "seconds": round(time.perf_counter() - started, 1),
    }
    import json
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
