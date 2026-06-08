"""
ComfyUI-MeshScript — LLM generation nodes.

    MeshScriptLLMLoader     Load a HuggingFace causal-LM model via transformers
    MeshScriptLLMGen        Text prompt → MeshScript  (txt2mesh)
    MeshScriptLLMRevise     Script + edit instruction → revised MeshScript (mesh2mesh)

Requires:  transformers  accelerate
           (torch is already present on Graydient — device_map="auto" uses the GPU)

Model dir: HF cache is routed to {ComfyUI}/models/llm/hf_cache/ so downloads persist
           between Graydient runs.  Default model: Qwen/Qwen2.5-Coder-7B-Instruct
"""

import os
import re

import folder_paths
from .nodes import _MS_ROOT

# ── lazy imports ──────────────────────────────────────────────────────────────

_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[ComfyUI-MeshScript] transformers/torch not available — "
          "LLM nodes will error at runtime. "
          "Add 'transformers' and 'accelerate' to requirements.pip.")

# ── persistent HF cache inside models dir ────────────────────────────────────

_LLM_DIR = os.path.join(folder_paths.models_dir, "llm")
os.makedirs(_LLM_DIR, exist_ok=True)
_HF_CACHE = os.path.join(_LLM_DIR, "hf_cache")

# ── context levels ────────────────────────────────────────────────────────────

CONTEXT_LEVELS = ["base", "+patterns", "+reference", "+all"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _system_prompt(context_level: str) -> str:
    """Build system prompt by reading meshscript docs at the requested depth."""
    if _MS_ROOT is None:
        return "You are a procedural 3D modeller. Write MeshScript programs."

    def _read(rel):
        p = os.path.join(_MS_ROOT, rel)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

    parts = [_read("prompt/system-prompt.md")]

    if "+patterns" in context_level or "+all" in context_level:
        pat = _read("docs/patterns.md")
        if pat:
            parts.append("\n---\n## Pattern Library\n\n" + pat)

    if "+reference" in context_level or "+all" in context_level:
        ref = _read("docs/op-reference.md")
        if ref:
            parts.append("\n---\n## Full Op Reference\n\n" + ref)

    return "\n".join(p for p in parts if p)


def _extract_script(text: str) -> str:
    """Strip <think> blocks (reasoning models), then pull first ```python fence."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:python|meshscript)\s*\n([\s\S]+?)```", text)
    return m.group(1).strip() if m else ""


def _chat_complete(model_pack, messages: list, temperature: float,
                   max_tokens: int) -> str:
    """Run a chat completion using transformers generate."""
    model, tokenizer = model_pack

    # Apply the model's chat template (handles system/user/assistant roles)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens   = max_tokens,
        pad_token_id     = tokenizer.eos_token_id,
        eos_token_id     = tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Return only the newly generated tokens
    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Node: MeshScriptLLMLoader ─────────────────────────────────────────────────

class MeshScriptLLMLoader:
    """
    Load a HuggingFace causal-LM for MeshScript generation.
    The model is downloaded on first use and cached in {ComfyUI}/models/llm/hf_cache/
    so subsequent runs skip the download.

    Default: Qwen/Qwen2.5-Coder-7B-Instruct  (~14 GB, bfloat16, runs on RTX 4090/5090)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default":  "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "tooltip":  "HuggingFace model ID or local path",
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "bfloat16 recommended — ~14 GB VRAM for 7B",
                }),
            }
        }

    RETURN_TYPES  = ("MS_LLM",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load"
    CATEGORY      = "MeshScript"

    def load(self, model_id: str, dtype: str):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is not installed.\n"
                "Add 'transformers' and 'accelerate' to requirements.pip."
            )

        # Route HF downloads to the persistent models directory
        os.environ["HF_HOME"] = _HF_CACHE

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }

        print(f"[MeshScriptLLMLoader] loading {model_id!r} ({dtype})")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype   = dtype_map[dtype],
            device_map    = "auto",   # places layers on GPU automatically
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"[MeshScriptLLMLoader] loaded — device: {device}")
        return ((model, tokenizer),)


# ── Node: MeshScriptLLMGen ────────────────────────────────────────────────────

class MeshScriptLLMGen:
    """
    Generate a MeshScript from a natural-language description.
    Used in the txt2mesh workflow.

    Outputs both the generated script and the original prompt (as 'spec')
    so the spec travels through the graph to SaveMeshWithScript.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MS_LLM",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "a coffee mug",
                    "tooltip":   "Natural-language description of the object to build",
                }),
                "context_level": (CONTEXT_LEVELS, {"default": "base"}),
                "temperature":   ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens":    ("INT",   {"default": 2048, "min": 256, "max": 8192}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("script",  "spec")
    FUNCTION      = "generate"
    CATEGORY      = "MeshScript"

    def generate(self, model, prompt: str, context_level: str,
                 temperature: float, max_tokens: int):
        sys_prompt = _system_prompt(context_level)
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    f"Design spec: {prompt}\n\n"
                    "Write a complete MeshScript program that constructs this object.\n"
                    "- Wrap the code in ```python fences.\n"
                    "- Use show(mesh, name) after each major component.\n"
                    "- Call ground(mesh) on the final result before the last show()."
                ),
            },
        ]
        raw    = _chat_complete(model, messages, temperature, max_tokens)
        script = _extract_script(raw)
        if not script:
            raise RuntimeError(
                f"LLM did not produce a fenced code block.\nRaw response:\n{raw[:500]}"
            )
        print(f"[MeshScriptLLMGen] generated {len(script)} chars")
        return (script, prompt)


# ── Node: MeshScriptLLMRevise ─────────────────────────────────────────────────

class MeshScriptLLMRevise:
    """
    Revise an existing MeshScript based on an edit instruction.
    Used in the mesh2mesh workflow.

    Inputs script + spec from LoadMeshWithScript (via links).
    edit_prompt is the user-provided edit instruction (field-mapped from 'prompt').
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":       ("MS_LLM",),
                "script":      ("STRING", {"forceInput": True,
                                           "tooltip": "Connect from LoadMeshWithScript.script"}),
                "spec":        ("STRING", {"forceInput": True,
                                           "tooltip": "Connect from LoadMeshWithScript.spec"}),
                "edit_prompt": ("STRING", {
                    "multiline": True,
                    "default":   "make the handle thinner",
                    "tooltip":   "Edit instruction — what to change",
                }),
                "context_level": (CONTEXT_LEVELS, {"default": "base"}),
                "temperature":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens":    ("INT",   {"default": 2048, "min": 256, "max": 8192}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("script",  "spec")
    FUNCTION      = "revise"
    CATEGORY      = "MeshScript"

    def revise(self, model, script: str, spec: str, edit_prompt: str,
               context_level: str, temperature: float, max_tokens: int):
        sys_prompt = _system_prompt(context_level)
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    f"Original spec: {spec}\n\n"
                    "Here is the existing MeshScript:\n"
                    f"```python\n{script}\n```\n\n"
                    f"Edit instruction: {edit_prompt}\n\n"
                    "Revise the script to apply the edit. "
                    "Preserve unchanged parts exactly. "
                    "Wrap the complete revised script in ```python fences."
                ),
            },
        ]
        raw        = _chat_complete(model, messages, temperature, max_tokens)
        new_script = _extract_script(raw)
        if not new_script:
            raise RuntimeError(
                f"LLM did not produce a fenced code block.\nRaw response:\n{raw[:500]}"
            )
        print(f"[MeshScriptLLMRevise] revised script: {len(new_script)} chars")
        return (new_script, edit_prompt)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MeshScriptLLMLoader":  MeshScriptLLMLoader,
    "MeshScriptLLMGen":     MeshScriptLLMGen,
    "MeshScriptLLMRevise":  MeshScriptLLMRevise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshScriptLLMLoader":  "MeshScript LLM Loader",
    "MeshScriptLLMGen":     "MeshScript LLM Gen",
    "MeshScriptLLMRevise":  "MeshScript LLM Revise",
}
