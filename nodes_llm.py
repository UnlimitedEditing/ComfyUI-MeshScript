"""
ComfyUI-MeshScript — LLM generation nodes.

    MeshScriptLLMLoader     Load a GGUF model via llama_cpp
    MeshScriptLLMGen        Text prompt → MeshScript  (txt2mesh)
    MeshScriptLLMRevise     Script + edit instruction → revised MeshScript (mesh2mesh)

Requires:  llama-cpp-python  (GPU: install with matching CUDA wheel — see pre_install_script)
Model dir: {ComfyUI}/models/llm/   (*.gguf files)
"""

import os
import re
import sys

import folder_paths

from .nodes import _MS_ROOT

# ── llama_cpp lazy import ─────────────────────────────────────────────────────

_LLAMA_AVAILABLE = False
try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    print("[ComfyUI-MeshScript] llama_cpp not installed — LLM nodes will error at runtime. "
          "Install via pre_install_script or: pip install llama-cpp-python")

# ── register models/llm folder ───────────────────────────────────────────────

_LLM_DIR = os.path.join(folder_paths.models_dir, "llm")
os.makedirs(_LLM_DIR, exist_ok=True)
if "llm" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("llm", _LLM_DIR)

# ── context level constants ───────────────────────────────────────────────────

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
    """Strip <think> blocks (Qwen3/reasoning models), then pull first ```python fence."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:python|meshscript)\s*\n([\s\S]+?)```", text)
    return m.group(1).strip() if m else ""


def _run_llm(llm, messages: list, temperature: float, max_tokens: int) -> str:
    resp = llm.create_chat_completion(
        messages   = messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    return resp["choices"][0]["message"]["content"]


# ── Node: MeshScriptLLMLoader ─────────────────────────────────────────────────

class MeshScriptLLMLoader:
    """
    Load a GGUF LLM model for MeshScript generation.
    Place *.gguf files in  {ComfyUI}/models/llm/
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_filename": ("STRING", {
                    "default": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                    "tooltip": "GGUF filename inside {ComfyUI}/models/llm/",
                }),
                "n_ctx": ("INT", {
                    "default": 4096, "min": 512, "max": 32768, "step": 512,
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1, "min": -1, "max": 200,
                    "tooltip": "-1 = all layers on GPU",
                }),
            }
        }

    RETURN_TYPES  = ("MS_LLM",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load"
    CATEGORY      = "MeshScript"

    def load(self, model_filename: str, n_ctx: int, n_gpu_layers: int):
        if not _LLAMA_AVAILABLE:
            raise RuntimeError("llama_cpp is not installed — cannot load model.")

        model_path = os.path.join(_LLM_DIR, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Place a GGUF file in {_LLM_DIR}"
            )

        print(f"[MeshScriptLLMLoader] loading {model_filename} "
              f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")
        llm = Llama(
            model_path    = model_path,
            n_ctx         = n_ctx,
            n_gpu_layers  = n_gpu_layers,
            verbose       = False,
        )
        return (llm,)


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
                "model":         ("MS_LLM",),
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
        raw    = _run_llm(model, messages, temperature, max_tokens)
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
        raw    = _run_llm(model, messages, temperature, max_tokens)
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
