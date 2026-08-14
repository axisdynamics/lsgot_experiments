"""
Extractor de hidden states directos via transformers.

A diferencia del modo API, aquí capturamos los vectores internos del modelo
en cada paso de generación — esto es lo que LSGOT llama "trajectory in ℌ".

Requiere modelo en formato HuggingFace (no GGUF).

Uso:
    extractor = HiddenStateExtractor("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    trajectory = extractor.extract(prompt, system_prompt)
    # trajectory.embeddings[i] = hidden state del último token en el paso i, capa final
"""

import os
import pickle
import hashlib
import numpy as np
import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class Trajectory:
    """Trayectoria completa de una generación (base class)."""
    prompt_id: int
    prompt_text: str
    response_text: str
    embeddings: List[np.ndarray]
    group: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.embeddings)

    @property
    def embedding_matrix(self) -> np.ndarray:
        if not self.embeddings:
            return np.array([])
        return np.vstack(self.embeddings)


@dataclass
class HiddenStateTrajectory(Trajectory):
    """Trayectoria con hidden states reales del modelo."""
    layer_idx: int = -1       # Capa de la que se extrajeron los hidden states
    n_layers: int = 0         # Total de capas del modelo
    token_ids: List[int] = field(default_factory=list)

    @property
    def is_real_latent(self) -> bool:
        """True si los embeddings son hidden states reales (no proxies semánticos)."""
        return True


class HiddenStateExtractor:
    """
    Extrae hidden states del modelo durante inferencia autoregresiva.

    Estrategia:
    - Genera tokens de a uno (greedy / temperatura)
    - En cada paso captura el hidden state de la última capa, último token
    - Esto forma la trayectoria real en ℌ

    Nota de memoria: el modelo 7B en bf16 ocupa ~15GB VRAM/RAM.
    En CPU es viable pero lento (~5-10 tok/s en CPU moderno).
    """

    def __init__(self, model_path: str,
                 layer_idx: int = -1,
                 device: str = "auto",
                 cache_dir: str = "cache_hidden/",
                 max_new_tokens: int = 512,
                 min_vram_gb: float = 20.0):
        self.model_path = model_path
        self.layer_idx = layer_idx  # -1 = última capa
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.min_vram_gb = min_vram_gb   # VRAM mínima para cargar en GPU
        os.makedirs(cache_dir, exist_ok=True)

        self._model = None
        self._tokenizer = None
        self._eos_ids: set = set()       # set de token IDs que señalan fin de generación

        # Elegir device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"HiddenStateExtractor — device: {self.device}, capa: {layer_idx}, min_vram: {min_vram_gb}GB")

    def load_model(self):
        """Carga el modelo en memoria (lazy, solo una vez)."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Cargando modelo desde {self.model_path}...")
        dtype = torch.bfloat16 if self.device in ("cuda", "cpu") else torch.float32

        # Gemma 4 compatibility: tokenizer_config.json stores extra_special_tokens
        # as a list, but transformers expects a dict. Patch in-place before loading.
        import json as _json
        _tcfg_path = os.path.join(self.model_path, "tokenizer_config.json")
        if os.path.exists(_tcfg_path):
            with open(_tcfg_path) as _f:
                _tcfg = _json.load(_f)
            if isinstance(_tcfg.get("extra_special_tokens"), list):
                _tcfg["extra_special_tokens"] = {}
                with open(_tcfg_path, "w") as _f:
                    _json.dump(_tcfg, _f, indent=2)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Gemma 4 tokenizer may ship without chat_template set.
        # Correct Gemma 4 instruct format: system prompt is prepended inside the
        # first user turn (NOT as a separate <start_of_turn>system turn).
        # Using Jinja2 namespace so the system content can be captured across loop iterations.
        if not getattr(self._tokenizer, "chat_template", None):
            self._tokenizer.chat_template = (
                "{{ bos_token }}"
                "{% set ns = namespace(system='') %}"
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{% set ns.system = message['content'] %}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<start_of_turn>user\n' }}"
                "{% if ns.system %}{{ ns.system | trim }}{{ '\n\n' }}{% set ns.system = '' %}{% endif %}"
                "{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}"
                "{% elif message['role'] == 'model' or message['role'] == 'assistant' %}"
                "{{ '<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
            )

        load_kwargs = dict(
            dtype=dtype,
            trust_remote_code=True,
            # "eager" evita flash-attention y kernels CUDA especializados que pueden
            # fallar en GPUs con compute capability no soportada por el build de PyTorch.
            attn_implementation="eager",
        )

        # Selección de device_map según VRAM disponible vs min_vram_gb del modelo.
        # Ejemplos: 7B bf16 ≈ 15GB → min_vram_gb=20; 2B bf16 ≈ 5GB → min_vram_gb=6.
        if self.device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= self.min_vram_gb:
                load_kwargs["device_map"] = "cuda:0"
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"  VRAM disponible: {vram_gb:.1f}GB ≥ {self.min_vram_gb}GB → cargando en GPU")
            else:
                load_kwargs["device_map"] = "cpu"
                self.device = "cpu"
                print(f"  VRAM disponible: {vram_gb:.1f}GB < {self.min_vram_gb}GB requeridos → cargando en CPU")
        else:
            load_kwargs["device_map"] = "cpu"
            self.device = "cpu"

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "CUDA" in str(e):
                # Incompatibilidad de compute capability GPU / build de PyTorch.
                # Fallback automático a CPU.
                print(f"\n  CUDA error al cargar modelo: {e}")
                print(f"  → Reintentando en CPU (inferencia más lenta)...")
                load_kwargs["device_map"] = "cpu"
                load_kwargs.pop("low_cpu_mem_usage", None)
                self.device = "cpu"
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **load_kwargs
                )
            else:
                raise

        self._model.eval()
        # Gemma 4 is multimodal: num_hidden_layers lives in text_config, not root config.
        _cfg = self._model.config
        if hasattr(_cfg, "num_hidden_layers"):
            n_layers = _cfg.num_hidden_layers
        elif hasattr(_cfg, "text_config") and hasattr(_cfg.text_config, "num_hidden_layers"):
            n_layers = _cfg.text_config.num_hidden_layers
        else:
            n_layers = getattr(_cfg, "num_layers", getattr(_cfg, "n_layer", 26))
            print(f"  Advertencia: num_hidden_layers no en config raíz, usando {n_layers}")

        # Calcular el conjunto de token IDs que terminan la generación.
        # Gemma IT usa <end_of_turn> además de <eos>. El patch de extra_special_tokens
        # puede desregistrar estos tokens, haciendo que convert_tokens_to_ids devuelva
        # unk_token_id. Se busca directamente en el vocabulario para evitar eso.
        eos_ids = set()
        if self._tokenizer.eos_token_id is not None:
            eos_ids.add(self._tokenizer.eos_token_id)
        vocab = self._tokenizer.get_vocab()
        for special_tok in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<eos>"):
            if special_tok in vocab:
                eos_ids.add(vocab[special_tok])
        self._eos_ids = eos_ids
        print(f"Modelo cargado — {n_layers} capas, device: {self.device}, EOS ids: {eos_ids}")
        return n_layers

    def extract(self, prompt: str, system_prompt: str,
                group: str = "") -> HiddenStateTrajectory:
        """
        Genera respuesta capturando hidden state por cada token generado.

        Returns:
            HiddenStateTrajectory con embeddings[i] = hidden state del token i
        """
        cache_key = self._cache_key(prompt, system_prompt)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        n_layers = self.load_model()

        # Construir input con chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback para modelos que no soportan el rol "system" en la plantilla
                # (ej. algunas versiones de Gemma base): prepend al mensaje de usuario.
                user_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                input_text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False, add_generation_prompt=True
                )
        else:
            input_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        input_ids = self._tokenizer(input_text, return_tensors="pt").input_ids

        # Safety valve: truncar solo si el input es absurdamente largo.
        # E4B-it (8B real, hidden=2560, 42 capas) en 20 GB VRAM:
        #   weights ≈ 16 GB + KV-cache(7000×42×10KB) ≈ 3 GB → total ~19 GB
        max_input_tokens = 7000
        if input_ids.shape[1] > max_input_tokens:
            print(f"  [truncar] input {input_ids.shape[1]} → {max_input_tokens} tokens "
                  f"(supera límite de seguridad para 24 GB VRAM)")
            input_ids = input_ids[:, -max_input_tokens:]

        # Con device_map="auto" el modelo puede estar distribuido;
        # movemos input_ids al device de la primera capa del modelo.
        first_device = next(self._model.parameters()).device
        input_ids = input_ids.to(first_device)

        hidden_states_sequence = []
        generated_token_ids = []
        generated_text_tokens = []

        # Generación token a token para capturar hidden states en cada paso
        past_key_values = None
        current_ids = input_ids.to(first_device)

        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self._model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )

                # Hidden state: capa layer_idx, último token — extraer y mover a CPU
                # antes de liberar outputs para minimizar pico de VRAM.
                token_hs = outputs.hidden_states[self.layer_idx][0, -1, :].float().cpu().numpy()
                hidden_states_sequence.append(token_hs)

                # Siguiente token (greedy) — solo el último logit, no toda la secuencia
                next_token_id = outputs.logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                token_id = next_token_id.item()
                generated_token_ids.append(token_id)

                # Guardar KV cache, liberar todo lo demás inmediatamente.
                # En el primer paso (prompt completo), outputs es muy grande — liberarlo
                # antes de la siguiente iteración evita el OOM en prompts largos (AXIS_DNA).
                past_key_values = outputs.past_key_values
                del outputs
                if step == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

                # Decodificar token
                token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
                generated_text_tokens.append(token_text)

                # EOS check — cubre <eos>, <end_of_turn> (Gemma), <|im_end|> (Qwen)
                if token_id in self._eos_ids:
                    break

                current_ids = next_token_id.to(first_device)

        response_text = self._tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Filtrar bloques <think> del texto de respuesta (deepseek-r1, no afecta a otros)
        import re
        clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        if not clean_response:
            clean_response = response_text

        prompt_id = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % (10**8)

        trajectory = HiddenStateTrajectory(
            prompt_id=prompt_id,
            prompt_text=prompt,
            response_text=clean_response,
            embeddings=hidden_states_sequence,
            group=group,
            layer_idx=self.layer_idx,
            n_layers=n_layers or 0,
            token_ids=generated_token_ids,
            metadata={
                "model": self.model_path,
                "mode": "hidden_states",
                "layer_idx": self.layer_idx,
                "n_tokens_generated": len(hidden_states_sequence),
                "device": self.device
            }
        )

        self._save_cache(cache_key, trajectory)
        return trajectory

    def extract_multilayer(self, prompt: str, system_prompt: str,
                            layers: List[int] = None) -> Dict[int, HiddenStateTrajectory]:
        """
        Extrae hidden states de múltiples capas simultáneamente en un solo forward pass.
        Útil para comparar curvatura en capas tempranas vs tardías.

        Args:
            layers: Índices absolutos de capas (ej. [7, 14, 20, 28, 35]).
                    Si None, usa cuartiles del modelo.
        """
        n_layers = self.load_model()
        _cfg = self._model.config
        if hasattr(_cfg, "num_hidden_layers"):
            n_layers = _cfg.num_hidden_layers
        elif hasattr(_cfg, "text_config") and hasattr(_cfg.text_config, "num_hidden_layers"):
            n_layers = _cfg.text_config.num_hidden_layers
        else:
            n_layers = getattr(_cfg, "num_layers", getattr(_cfg, "n_layer", 26))

        if layers is None:
            layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, -1]

        # Cache check — evita re-inferencia si ya existe
        cache_key = self._cache_key_multilayer(prompt, system_prompt, layers)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # Construir input con chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                user_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                input_text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False, add_generation_prompt=True
                )
        else:
            input_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        input_ids = self._tokenizer(input_text, return_tensors="pt").input_ids

        # Truncación de seguridad (igual que extract())
        max_input_tokens = 7000
        if input_ids.shape[1] > max_input_tokens:
            print(f"  [truncar] input {input_ids.shape[1]} → {max_input_tokens} tokens "
                  f"(límite para E4B-it en 20 GB VRAM)")
            input_ids = input_ids[:, -max_input_tokens:]

        first_device = next(self._model.parameters()).device
        input_ids = input_ids.to(first_device)

        hs_by_layer = {l: [] for l in layers}
        token_ids = []
        past_key_values = None
        current_ids = input_ids

        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self._model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )

                # Capturar hidden states de todas las capas objetivo antes de liberar outputs
                for l in layers:
                    hs = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                    hs_by_layer[l].append(hs)

                next_token_id = outputs.logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                tid = next_token_id.item()
                token_ids.append(tid)

                # Guardar KV cache y liberar outputs inmediatamente para evitar OOM
                past_key_values = outputs.past_key_values
                del outputs
                if step == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

                if tid in self._eos_ids:
                    break

                current_ids = next_token_id.to(first_device)

        response_text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        import re
        clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        if not clean_response:
            clean_response = response_text

        prompt_id = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % (10**8)

        trajectories = {}
        for l in layers:
            trajectories[l] = HiddenStateTrajectory(
                prompt_id=prompt_id,
                prompt_text=prompt,
                response_text=clean_response,
                embeddings=hs_by_layer[l],
                group="",
                layer_idx=l,
                n_layers=n_layers,
                token_ids=token_ids
            )

        self._save_cache(cache_key, trajectories)
        return trajectories

    def _cache_key_multilayer(self, prompt: str, system_prompt: str, layers: List[int]) -> str:
        layers_str = ",".join(str(l) for l in sorted(str(x) for x in layers))
        content = (f"hs_ml||{self._CACHE_VERSION}||{self.model_path}||"
                   f"{layers_str}||{prompt}||{system_prompt}")
        return hashlib.md5(content.encode()).hexdigest()

    # Increment when extraction logic changes to auto-invalidate old cache entries.
    _CACHE_VERSION = "v4"  # v4: cache key usa system_prompt completo (fix colisión axis/axis_nowit)

    def _cache_key(self, prompt: str, system_prompt: str) -> str:
        content = (f"hs||{self._CACHE_VERSION}||{self.model_path}||"
                   f"{self.layer_idx}||{prompt}||{system_prompt}")
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[HiddenStateTrajectory]:
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, trajectory: HiddenStateTrajectory):
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(trajectory, f)


def extract_group_hidden(group_name: str, group_config: Dict,
                          prompts: List[Dict],
                          model_path: str,
                          cache_dir: str = "cache_hidden/",
                          max_new_tokens: int = 512) -> List[HiddenStateTrajectory]:
    """
    Extrae trayectorias de hidden states para un grupo completo.
    Reutiliza el mismo extractor (modelo cargado una vez).
    """
    extractor = HiddenStateExtractor(
        model_path=model_path,
        cache_dir=cache_dir,
        max_new_tokens=max_new_tokens
    )

    # System prompt
    if 'system_prompt_path' in group_config:
        with open(group_config['system_prompt_path'], 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    else:
        system_prompt = group_config.get('system_prompt', "You are a helpful assistant.")

    trajectories = []
    for prompt_data in tqdm(prompts, desc=f"[hidden_states] {group_name}"):
        prompt_text = prompt_data.get('text', prompt_data.get('question', ''))
        traj = extractor.extract(prompt_text, system_prompt, group=group_name)
        trajectories.append(traj)

    return trajectories
