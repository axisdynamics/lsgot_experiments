#!/usr/bin/env python3
"""
Verificador de longitud de prompts en tokens reales.
Usa los tokenizadores de cada modelo del panel para reportar
conteos exactos. Sin esto, el confound de longitud sigue abierto.

Uso:
  python verify_tokens.py                    # todos los prompts
  python verify_tokens.py --model gemma4_31b # solo un modelo
"""

import sys
import os
from pathlib import Path
from argparse import ArgumentParser

BASE = Path(__file__).parent

# Prompts a verificar (relativo a BASE)
# generic_long MIA: emparejado a MIA axis (Gemma4: 2,105 vs 2,110 tokens)
# generic_long SIA: emparejado a SIA axis (Gemma4: 3,957 vs 3,945 tokens)
PROMPT_PATHS = {
    "MIA_axis":           "mia/prompts/axis.dna",
    "MIA_generic_long":   "mia/prompts/generic_long.txt",
    "MIA_generic_short":  "mia/prompts/generic_short.txt",
    "SIA_axis":           "sia/prompts/axis.dna",
    "SIA_generic_long":   "sia/prompts/generic_long.txt",
    "SIA_generic_short":  "sia/prompts/generic_short.txt",
    "vanilla":            None,  # "You are a helpful assistant."
}

# Modelos del panel con sus tokenizadores HF
MODELS = {
    "gemma4_e4b":     "google/gemma-4-E4B-it",
    "gemma4_31b":     "google/gemma-4-31B-it",
    "deepseek_r1_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen25_7b":      "Qwen/Qwen2.5-7B-Instruct",
    "deepseek_r1_32b":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

VANILLA_TEXT = "You are a helpful assistant."


def load_prompt(path_str: str | None) -> str:
    if path_str is None:
        return VANILLA_TEXT
    path = BASE / path_str
    if not path.exists():
        return f"[MISSING: {path}]"
    with open(path, encoding="utf-8") as f:
        return f.read()


def count_tokens(text: str, tokenizer) -> int:
    """Cuenta tokens reales usando el tokenizador HF."""
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return int(ids.shape[1])


def main():
    parser = ArgumentParser(description="Verificar tokens reales de prompts")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Modelo específico (default: todos)")
    parser.add_argument("--chat-template", action="store_true",
                        help="Contar con chat_template aplicado (system+user)")
    parser.add_argument("--token", default=None, help="HF token")
    args = parser.parse_args()

    # Cargar todos los textos de prompts
    prompts = {name: load_prompt(path) for name, path in PROMPT_PATHS.items()}
    print(f"Prompts cargados:")
    for name, text in prompts.items():
        print(f"  {name:25s}: {len(text):>6,} chars | {len(text.split()):>5} words")
    print()

    models_to_check = [args.model] if args.model else list(MODELS.keys())

    for model_key in models_to_check:
        hf_id = MODELS[model_key]
        print(f"{'='*70}")
        print(f"Modelo: {hf_id}")
        print(f"{'='*70}")

        try:
            from transformers import AutoTokenizer
            tok_kwargs = {"trust_remote_code": True}
            if args.token:
                tok_kwargs["token"] = args.token
            tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kwargs)
        except Exception as e:
            print(f"  ERROR cargando tokenizador: {e}")
            print(f"  Skip — ¿modelo no descargado?")
            continue

        print(f"  Vocab size: {tokenizer.vocab_size:,}")
        print()

        header = f"{'Prompt':25s} {'chars':>7} {'tokens':>7} {'ratio':>7} {'%axis':>7}"
        print(header)
        print("-" * len(header))

        # Encontrar axis de referencia (MIA o SIA según lo que exista)
        axis_names = [n for n in prompts if "axis" in n.lower()]
        axis_tokens = {}
        for aname in axis_names:
            axis_tokens[aname] = count_tokens(prompts[aname], tokenizer)

        for name, text in prompts.items():
            if "[MISSING" in text:
                print(f"  {name:25s} {'—':>7} {'—':>7}")
                continue
            n_tok = count_tokens(text, tokenizer)
            ratio = len(text) / max(n_tok, 1)

            # % relativo al axis correspondiente
            if "MIA" in name:
                ref = axis_tokens.get("MIA_axis")
            elif "SIA" in name:
                ref = axis_tokens.get("SIA_axis")
            else:
                ref = axis_tokens.get("MIA_axis")  # default
            pct = f"{n_tok/ref*100:.0f}%" if ref else "—"

            print(f"  {name:25s} {len(text):>7,} {n_tok:>7,} {ratio:>6.1f} {pct:>7}")

        # Comparaciones clave
        print()
        print("  Contrastes clave:")
        mia_ax = axis_tokens.get("MIA_axis", 0)
        sia_ax = axis_tokens.get("SIA_axis", 0)
        gl_tok = count_tokens(prompts.get("MIA_generic_long", ""), tokenizer)
        gs_tok = count_tokens(prompts.get("MIA_generic_short", ""), tokenizer)
        van_tok = count_tokens(prompts["vanilla"], tokenizer) if "vanilla" in prompts else 0

        if mia_ax and gl_tok:
            delta = abs(gl_tok - mia_ax) / mia_ax * 100
            icon = "✓" if delta < 5 else ("⚠" if delta < 15 else "✗")
            print(f"  {icon} generic_long vs MIA_axis: {gl_tok:,} vs {mia_ax:,} tokens (Δ={delta:.1f}%)")
        if mia_ax and gs_tok:
            print(f"    generic_short vs MIA_axis: {gs_tok:,} vs {mia_ax:,} tokens (Δ={abs(gs_tok-mia_ax)/mia_ax*100:.1f}%)")
        if sia_ax and gl_tok:
            delta_sia = abs(gl_tok - sia_ax) / sia_ax * 100
            icon_sia = "✓" if delta_sia < 5 else ("⚠" if delta_sia < 15 else "✗")
            print(f"  {icon_sia} generic_long vs SIA_axis: {gl_tok:,} vs {sia_ax:,} tokens (Δ={delta_sia:.1f}%)")
        if gl_tok and gs_tok:
            print(f"    Long/Short ratio: {gl_tok/gs_tok:.2f}x")

        print()

    print("Hecho. Si algún Δ% > 5%, ajustar generic_long.txt y re-ejecutar.")
    print("Target: generic_long dentro de ±3% de axis.dna en tokens.")


if __name__ == "__main__":
    main()
