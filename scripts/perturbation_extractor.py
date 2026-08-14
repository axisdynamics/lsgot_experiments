"""
Perturbation Extractor — H4_rev

Injecta ruido gaussiano en hidden states de capas específicas durante la generación
token a token, vía PyTorch forward hooks. Mide si el grupo axis recupera su firma
geométrica más rápido que vanilla/generic (discriminante resiliencia dependiente
de contenido vs resonancia pura).

Arquitectura target: google/gemma-4-E4B-it (42 capas, hidden_dim=2560)
  - L21 (idx=20): umbral topológico (W₁ máximo en layer analysis MIA)
  - L28 (idx=27): pico de concentración geométrica (SampEn mínimo)
"""

import os
import pickle
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_CACHE_VERSION = "perturb_v2"


# ── Resultado de una trayectoria ──────────────────────────────────────────────

@dataclass
class Trajectory:
    embeddings: np.ndarray   # (T, D) float32
    n_steps: int
    group: str = ""
    t_inj: Optional[int] = None
    sigma: Optional[float] = None
    perturbed_layers: List[int] = field(default_factory=list)

    @property
    def embedding_matrix(self) -> np.ndarray:
        return self.embeddings


# ── Detección robusta de capas decoder ───────────────────────────────────────

def _get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """
    Encuentra las capas decoder del transformer probando rutas conocidas.

    Rutas en orden de prioridad:
      Gemma4ForConditionalGeneration : model.model.language_model.layers
      Gemma/Qwen/DeepSeek CausalLM   : model.model.layers
      GPT-2 / Falcon                  : model.transformer.h
      Fallback                        : búsqueda de ModuleList más largo
    """
    # Gemma4: ForConditionalGeneration → Gemma4Model → language_model → layers
    try:
        layers = model.model.language_model.layers
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers
    except AttributeError:
        pass

    # CausalLM estándar: model.model.layers
    try:
        layers = model.model.layers
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers
    except AttributeError:
        pass

    # GPT-2 / Falcon
    try:
        layers = model.transformer.h
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers
    except AttributeError:
        pass

    # Fallback: buscar el ModuleList más largo en los primeros 3 niveles
    best = _find_deepest_modulelist(model, max_depth=3)
    if best is not None:
        print(f"  [warn] Capas encontradas via fallback: {type(best).__name__} len={len(best)}")
        return best

    raise AttributeError(
        f"No se puede acceder a las capas del modelo ({type(model).__name__}).\n"
        "Rutas probadas: model.model.language_model.layers, model.model.layers, "
        "model.transformer.h.\n"
        "Inspecciona la arquitectura con: print(model)"
    )


def _find_deepest_modulelist(module: torch.nn.Module, max_depth: int,
                              _depth: int = 0) -> Optional[torch.nn.ModuleList]:
    """Busca recursivamente el ModuleList más largo hasta max_depth."""
    best = None
    best_len = 0
    for child in module.children():
        if isinstance(child, torch.nn.ModuleList) and len(child) > best_len:
            best = child
            best_len = len(child)
        if _depth < max_depth:
            candidate = _find_deepest_modulelist(child, max_depth, _depth + 1)
            if candidate is not None and len(candidate) > best_len:
                best = candidate
                best_len = len(candidate)
    return best


def _probe_config(model: torch.nn.Module) -> Tuple[int, int]:
    """Extrae (n_layers, hidden_dim) de la config tolerando Gemma4 multimodal."""
    cfg = model.config
    # Gemma4: atributos de texto bajo text_config
    text_cfg = getattr(cfg, "text_config", cfg)

    n_layers = (
        getattr(text_cfg, "num_hidden_layers", None)
        or getattr(cfg, "num_hidden_layers", None)
        or getattr(text_cfg, "num_decoder_layers", None)
        or getattr(cfg, "num_decoder_layers", None)
    )
    hidden_dim = (
        getattr(text_cfg, "hidden_size", None)
        or getattr(cfg, "hidden_size", None)
    )

    # Validación cruzada con las capas reales
    try:
        layers = _get_decoder_layers(model)
        if n_layers is None:
            n_layers = len(layers)
    except AttributeError:
        pass

    if n_layers is None or hidden_dim is None:
        raise RuntimeError(
            f"No se pudo leer n_layers ({n_layers}) / hidden_size ({hidden_dim}).\n"
            f"Config class: {type(cfg).__name__}\n"
            f"Config attrs: {[a for a in dir(cfg) if not a.startswith('_')]}"
        )
    return int(n_layers), int(hidden_dim)


# ── Hook manager ──────────────────────────────────────────────────────────────

class PerturbationHookManager:
    """
    Registra forward hooks en las capas decoder target.
    Inyecta ε ~ N(0, σ²·I) en output[0] cuando está activo.
    Se activa exactamente en t_inj durante UN paso y se desactiva inmediatamente.
    """

    def __init__(self, model: torch.nn.Module, layer_indices: List[int], sigma: float):
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.sigma = sigma
        self._active = False
        self._handles: List = []

    def _make_hook(self):
        def _hook(module, inputs, output):
            if not self._active:
                return output
            if isinstance(output, tuple):
                hs = output[0]
                noise = torch.randn_like(hs) * self.sigma
                return (hs + noise,) + output[1:]
            else:
                noise = torch.randn_like(output) * self.sigma
                return output + noise
        return _hook

    def register(self):
        layers = _get_decoder_layers(self.model)
        hook_fn = self._make_hook()
        for idx in self.layer_indices:
            if idx >= len(layers):
                raise IndexError(
                    f"Capa índice {idx} fuera de rango (modelo tiene {len(layers)} capas)"
                )
            h = layers[idx].register_forward_hook(hook_fn)
            self._handles.append(h)

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._active = False


# ── Extractor principal ───────────────────────────────────────────────────────

class PerturbationExtractor:

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        layer_idx: int = -1,
        max_input_tokens: int = 9000,
        cache_dir: str = "cache_perturb/",
        device: str = "cuda",
        min_vram_gb: float = 16.0,
        hf_token: Optional[str] = None,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.layer_idx = layer_idx
        self.max_input_tokens = max_input_tokens
        self.cache_dir = cache_dir
        self.device = device
        self.min_vram_gb = min_vram_gb
        self.hf_token = hf_token

        self.model = None
        self.tokenizer = None
        self._eos_ids: set = set()
        self._device_obj = None
        self._n_layers: Optional[int] = None

        os.makedirs(cache_dir, exist_ok=True)

    def load_model(self) -> Tuple[int, int]:
        load_kwargs: Dict = dict(trust_remote_code=True, attn_implementation="eager")
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        if self.device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= self.min_vram_gb:
                load_kwargs.update(device_map="cuda:0", dtype=torch.bfloat16,
                                   low_cpu_mem_usage=True)
                print(f"  VRAM disponible: {vram_gb:.1f}GB → BF16 en GPU")
            else:
                load_kwargs["device_map"] = "cpu"
                print(f"  VRAM insuficiente ({vram_gb:.1f}GB) → CPU float32")
        else:
            load_kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        self.model.eval()

        tok_kwargs: Dict = dict(trust_remote_code=True)
        if self.hf_token:
            tok_kwargs["token"] = self.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tok_kwargs)

        try:
            self._device_obj = next(self.model.parameters()).device
        except StopIteration:
            self._device_obj = torch.device("cpu")

        # EOS tokens
        for attr in ("eos_token_id", "pad_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                self._eos_ids.update(val)
            else:
                self._eos_ids.add(int(val))
        for tok_str in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<eos>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self._eos_ids.add(int(tid))

        n_layers, hidden_dim = _probe_config(self.model)
        self._n_layers = n_layers

        # Verificar que las capas son accesibles
        layers = _get_decoder_layers(self.model)
        print(f"  Capas decoder encontradas: {len(layers)} (config dice {n_layers})")
        print(f"  hidden_dim verificado: {hidden_dim}")
        print(f"  EOS tokens: {self._eos_ids}")
        return n_layers, hidden_dim

    def _tokenize(self, prompt: str, system_prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

        ids = self.tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] > self.max_input_tokens:
            ids = ids[:, :self.max_input_tokens]
        return ids.to(self._device_obj)

    def _generate(
        self,
        input_ids: torch.Tensor,
        hook_mgr: Optional[PerturbationHookManager] = None,
        t_inj: Optional[int] = None,
    ) -> np.ndarray:
        states: List[np.ndarray] = []
        past_kv = None
        current_ids = input_ids

        with torch.no_grad():
            for step in range(self.max_new_tokens):

                if hook_mgr is not None and t_inj is not None and step == t_inj:
                    hook_mgr.activate()

                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )

                if hook_mgr is not None and hook_mgr._active:
                    hook_mgr.deactivate()

                # Hidden states: accesibles directamente o bajo language_model_outputs
                hs_all = getattr(outputs, "hidden_states", None)
                if hs_all is None:
                    lm_out = getattr(outputs, "language_model_outputs", None)
                    if lm_out is not None:
                        hs_all = getattr(lm_out, "hidden_states", None)
                if hs_all is None:
                    raise RuntimeError(
                        "outputs.hidden_states es None. "
                        "Verifica que output_hidden_states=True sea soportado."
                    )

                hs = hs_all[self.layer_idx][0, -1, :]
                states.append(hs.float().cpu().numpy())

                # Logits
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    lm_out = getattr(outputs, "language_model_outputs", None)
                    if lm_out is not None:
                        logits = getattr(lm_out, "logits", None)
                if logits is None:
                    raise RuntimeError("No se encontraron logits en outputs.")

                next_tok = int(logits[0, -1, :].argmax())
                past_kv = outputs.past_key_values
                current_ids = torch.tensor(
                    [[next_tok]], dtype=torch.long, device=self._device_obj
                )
                del outputs

                if next_tok in self._eos_ids:
                    break

        return np.array(states, dtype=np.float32)

    def extract_baseline(self, prompt: str, system_prompt: str, group: str = "") -> Trajectory:
        key = self._cache_key("base", prompt, system_prompt, 0, 0.0, [])
        cached = self._load(key)
        if cached is not None:
            return Trajectory(embeddings=cached, n_steps=len(cached), group=group)
        ids = self._tokenize(prompt, system_prompt)
        emb = self._generate(ids)
        self._save(key, emb)
        return Trajectory(embeddings=emb, n_steps=len(emb), group=group)

    def extract_perturbed(
        self,
        prompt: str,
        system_prompt: str,
        group: str = "",
        t_inj: int = 50,
        sigma: float = 4.41,
        layer_indices: Optional[List[int]] = None,
    ) -> Trajectory:
        if layer_indices is None:
            layer_indices = [20]

        key = self._cache_key("perturb", prompt, system_prompt, t_inj, sigma, layer_indices)
        cached = self._load(key)
        if cached is not None:
            return Trajectory(
                embeddings=cached, n_steps=len(cached), group=group,
                t_inj=t_inj, sigma=sigma, perturbed_layers=layer_indices,
            )

        ids = self._tokenize(prompt, system_prompt)
        hook = PerturbationHookManager(self.model, layer_indices, sigma)
        hook.register()
        try:
            emb = self._generate(ids, hook_mgr=hook, t_inj=t_inj)
        finally:
            hook.remove()

        self._save(key, emb)
        return Trajectory(
            embeddings=emb, n_steps=len(emb), group=group,
            t_inj=t_inj, sigma=sigma, perturbed_layers=layer_indices,
        )

    def _cache_key(self, prefix, prompt, system_prompt, t_inj, sigma, layers):
        layers_str = "_".join(map(str, sorted(layers)))
        content = (
            f"{_CACHE_VERSION}||{prefix}||{self.model_path}||{self.layer_idx}"
            f"||{prompt}||{system_prompt}||t{t_inj}||s{sigma:.4f}||L{layers_str}"
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _load(self, key: str) -> Optional[np.ndarray]:
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save(self, key: str, data: np.ndarray):
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)


# ── Utilidades ────────────────────────────────────────────────────────────────

def compute_sigma(mean_velocity: float, hidden_dim: int, k: float = 1.0) -> float:
    """
    σ tal que ||ε||₂ ≈ k × mean_velocity.
    Ejemplo (Gemma4-E4B-it): mean_velocity=223.4, hidden_dim=2560, k=1.0 → σ ≈ 4.41
    """
    return k * mean_velocity / math.sqrt(hidden_dim)
