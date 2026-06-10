"""
ComfyUI-MeshScript — nodes_ir.py

Structured-output ("IR") generation nodes. Instead of asking the LLM to write
free-form CanvasScript/MeshScript and hoping the syntax and op usage are
valid, generation is constrained token-by-token to a JSON Schema (ir.schema)
via lm-format-enforcer. The resulting IR JSON is then semantically validated
(ir.validate) — checking $ref ordering and Document/Layer/Mesh type
compatibility between steps — with a bounded retry-with-feedback loop, and
finally compiled (ir.compile) into plain .cnv / .ms source text that plugs
into the existing CanvasScriptExecute / MeshScriptExecute nodes unchanged.

    CanvasScriptLLMGenIR    Text prompt -> CanvasScript IR -> .cnv script

Requires:  transformers  accelerate  lm-format-enforcer
"""

import json
import os

from .nodes import _MS_ROOT

_LMFE_AVAILABLE = False
try:
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    _LMFE_AVAILABLE = True
except ImportError:
    print("[ComfyUI-MeshScript] lm-format-enforcer/transformers/torch not available — "
          "IR generation nodes will error at runtime. "
          "Add 'lm-format-enforcer', 'transformers' and 'accelerate' to requirements.pip.")


_CANVAS_GENERATED_MARKER = (
    "<!-- BEGIN GENERATED: ir.dictionary_section(CANVAS_SPECS) -->\n"
    "<!-- appended at runtime by nodes_ir.py -->\n"
    "<!-- END GENERATED -->"
)

_MESH_GENERATED_MARKER = (
    "<!-- BEGIN GENERATED: ir.dictionary_section(MESH_SPECS) -->\n"
    "<!-- appended at runtime by nodes_ir.py -->\n"
    "<!-- END GENERATED -->"
)


def _canvas_ir_system_prompt() -> str:
    """Load prompt/canvas-ir-prompt.md and splice in the live op dictionary."""
    if _MS_ROOT is None:
        return "Respond with CanvasScript IR JSON only."

    from ir import CANVAS_SPECS, dictionary_section

    path = os.path.join(_MS_ROOT, "prompt", "canvas-ir-prompt.md")
    base = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    if _CANVAS_GENERATED_MARKER in base:
        base = base.replace(_CANVAS_GENERATED_MARKER, dictionary_section(CANVAS_SPECS))
    return base


def _mesh_ir_system_prompt() -> str:
    """Load prompt/mesh-ir-prompt.md and splice in the live op dictionary."""
    if _MS_ROOT is None:
        return "Respond with MeshScript IR JSON only."

    from ir import MESH_SPECS, dictionary_section

    path = os.path.join(_MS_ROOT, "prompt", "mesh-ir-prompt.md")
    base = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    if _MESH_GENERATED_MARKER in base:
        base = base.replace(_MESH_GENERATED_MARKER, dictionary_section(MESH_SPECS))
    return base


def _constrained_generate(model_pack, messages, schema, temperature, max_tokens) -> str:
    """Run a chat completion with generation constrained to `schema`."""
    model, tokenizer = model_pack

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    parser = JsonSchemaParser(schema)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    print(f"[_constrained_generate] prompt_tokens={inputs.input_ids.shape[1]}  "
          f"max_new_tokens={max_tokens}  temperature={temperature}")

    gen_kwargs = dict(
        max_new_tokens          = max_tokens,
        pad_token_id            = tokenizer.eos_token_id,
        eos_token_id            = tokenizer.eos_token_id,
        prefix_allowed_tokens_fn = prefix_fn,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)

    truncated = new_ids.shape[0] >= max_tokens
    print(f"[_constrained_generate] generated {new_ids.shape[0]} tokens ({len(raw)} chars)"
          + (" -- HIT max_new_tokens, output likely truncated" if truncated else ""))
    print(f"[_constrained_generate] -- raw IR --\n{raw}")

    return raw


# ── Node: CanvasScriptLLMGenIR ───────────────────────────────────────────────

class CanvasScriptLLMGenIR:
    """
    Generate a CanvasScript program from a natural-language description via a
    schema-constrained IR. The model can only emit JSON that conforms to
    ir.schema.build_schema(CANVAS_SPECS) — every op name, argument, and
    Document/Layer reference is enforced token-by-token. The resulting IR is
    then semantically validated and compiled to .cnv source.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MS_LLM",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "a poster for a midnight running event",
                    "tooltip":   "Natural-language description of the design to build",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens":  ("INT",   {"default": 2048, "min": 256, "max": 8192}),
                "max_retries": ("INT",   {"default": 2, "min": 0, "max": 5,
                                          "tooltip": "Extra generation attempts if the IR fails semantic validation"}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING")
    RETURN_NAMES  = ("script",  "spec",  "ir_json")
    FUNCTION      = "generate"
    CATEGORY      = "CanvasScript"

    def generate(self, model, prompt: str, temperature: float, max_tokens: int, max_retries: int):
        if not _LMFE_AVAILABLE:
            raise RuntimeError(
                "lm-format-enforcer is not installed.\n"
                "Add 'lm-format-enforcer', 'transformers' and 'accelerate' to requirements.pip."
            )
        if _MS_ROOT is None:
            raise RuntimeError("meshscript library not found — set MESHSCRIPT_PATH")

        from ir import CANVAS_SPECS, CANVAS_SPECS_BY_NAME, build_schema, validate, compile_ir

        print(f"[CanvasScriptLLMGenIR] received prompt ({len(prompt)} chars): {prompt!r}")
        print(f"[CanvasScriptLLMGenIR] temperature={temperature}  max_tokens={max_tokens}  "
              f"max_retries={max_retries}")

        schema = build_schema(CANVAS_SPECS)
        messages = [
            {"role": "system", "content": _canvas_ir_system_prompt()},
            {"role": "user", "content": (
                f"Design spec: {prompt}\n\n"
                "Respond with the IR JSON object for this design."
            )},
        ]

        ir_json_text = None
        last_errors  = None

        for attempt in range(max_retries + 1):
            print(f"[CanvasScriptLLMGenIR] attempt {attempt + 1}/{max_retries + 1}")
            raw = _constrained_generate(model, messages, schema, temperature, max_tokens)

            try:
                ir = json.loads(raw)
            except json.JSONDecodeError as e:
                last_errors = [f"output was not valid JSON: {e}"]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "That was not valid JSON. Respond again with ONLY the IR JSON object."})
                continue

            errors = validate(ir, CANVAS_SPECS_BY_NAME)
            if not errors:
                ir_json_text = raw
                break

            print(f"[CanvasScriptLLMGenIR] validation errors: {errors}")
            last_errors = errors
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                "The IR has the following errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nFix these and resend the complete IR JSON object."
            )})

        if ir_json_text is None:
            raise RuntimeError(
                f"LLM did not produce valid IR after {max_retries + 1} attempt(s).\n"
                "Last errors:\n" + "\n".join(f"- {e}" for e in (last_errors or []))
            )

        ir     = json.loads(ir_json_text)
        script = compile_ir(ir, CANVAS_SPECS_BY_NAME)
        print(f"[CanvasScriptLLMGenIR] compiled script ({len(script)} chars, "
              f"{script.count('show(')} show() calls):\n{script}")

        return (script, prompt, ir_json_text)


# ── Node: MeshScriptLLMGenIR ─────────────────────────────────────────────────

class MeshScriptLLMGenIR:
    """
    Generate a MeshScript program from a natural-language description via a
    schema-constrained IR. The model can only emit JSON that conforms to
    ir.schema.build_schema(MESH_SPECS) — every op name, argument, and
    Mesh/Profile/Path reference is enforced token-by-token. The resulting IR
    is then semantically validated and compiled to .ms source.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MS_LLM",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "a two-tier birthday cake",
                    "tooltip":   "Natural-language description of the object to model",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens":  ("INT",   {"default": 2048, "min": 256, "max": 8192}),
                "max_retries": ("INT",   {"default": 2, "min": 0, "max": 5,
                                          "tooltip": "Extra generation attempts if the IR fails semantic validation"}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING")
    RETURN_NAMES  = ("script",  "spec",  "ir_json")
    FUNCTION      = "generate"
    CATEGORY      = "MeshScript"

    def generate(self, model, prompt: str, temperature: float, max_tokens: int, max_retries: int):
        if not _LMFE_AVAILABLE:
            raise RuntimeError(
                "lm-format-enforcer is not installed.\n"
                "Add 'lm-format-enforcer', 'transformers' and 'accelerate' to requirements.pip."
            )
        if _MS_ROOT is None:
            raise RuntimeError("meshscript library not found — set MESHSCRIPT_PATH")

        from ir import MESH_SPECS, MESH_SPECS_BY_NAME, build_schema, validate, compile_ir

        print(f"[MeshScriptLLMGenIR] received prompt ({len(prompt)} chars): {prompt!r}")
        print(f"[MeshScriptLLMGenIR] temperature={temperature}  max_tokens={max_tokens}  "
              f"max_retries={max_retries}")

        schema = build_schema(MESH_SPECS)
        messages = [
            {"role": "system", "content": _mesh_ir_system_prompt()},
            {"role": "user", "content": (
                f"Design spec: {prompt}\n\n"
                "Respond with the IR JSON object for this design."
            )},
        ]

        ir_json_text = None
        last_errors  = None

        for attempt in range(max_retries + 1):
            print(f"[MeshScriptLLMGenIR] attempt {attempt + 1}/{max_retries + 1}")
            raw = _constrained_generate(model, messages, schema, temperature, max_tokens)

            try:
                ir = json.loads(raw)
            except json.JSONDecodeError as e:
                last_errors = [f"output was not valid JSON: {e}"]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "That was not valid JSON. Respond again with ONLY the IR JSON object."})
                continue

            errors = validate(ir, MESH_SPECS_BY_NAME)
            if not errors:
                ir_json_text = raw
                break

            print(f"[MeshScriptLLMGenIR] validation errors: {errors}")
            last_errors = errors
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                "The IR has the following errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nFix these and resend the complete IR JSON object."
            )})

        if ir_json_text is None:
            raise RuntimeError(
                f"LLM did not produce valid IR after {max_retries + 1} attempt(s).\n"
                "Last errors:\n" + "\n".join(f"- {e}" for e in (last_errors or []))
            )

        ir     = json.loads(ir_json_text)
        script = compile_ir(ir, MESH_SPECS_BY_NAME)
        print(f"[MeshScriptLLMGenIR] compiled script ({len(script)} chars, "
              f"{script.count('show(')} show() calls):\n{script}")

        return (script, prompt, ir_json_text)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "CanvasScriptLLMGenIR": CanvasScriptLLMGenIR,
    "MeshScriptLLMGenIR":   MeshScriptLLMGenIR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanvasScriptLLMGenIR": "CanvasScript LLM Gen (IR)",
    "MeshScriptLLMGenIR":   "MeshScript LLM Gen (IR)",
}
