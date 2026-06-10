"""
ComfyUI-MeshScript — nodes_ir.py

Structured-output ("IR") generation nodes. Instead of asking the LLM to write
free-form CanvasScript/MeshScript and hoping the syntax and op usage are
valid, generation is constrained token-by-token to a JSON Schema (ir.schema)
via xgrammar. The resulting IR JSON is then semantically validated
(ir.validate) — checking $ref ordering and Document/Layer/Mesh type
compatibility between steps — with a bounded retry-with-feedback loop, and
finally compiled (ir.compile) into plain .cnv / .ms source text that plugs
into the existing CanvasScriptExecute / MeshScriptExecute nodes unchanged.

    CanvasScriptLLMGenIR    Text prompt -> CanvasScript IR -> .cnv script

Requires:  transformers  accelerate  xgrammar
"""

import json
import os

from .nodes import _MS_ROOT

_XGR_AVAILABLE = False
_XGR_IMPORT_ERROR = None
try:
    import torch
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor as _XGRLogitsProcessor
    _XGR_AVAILABLE = True
except Exception as e:
    _XGR_IMPORT_ERROR = e
    import traceback
    print("[ComfyUI-MeshScript] xgrammar/transformers/torch not available — "
          "IR generation nodes will error at runtime. "
          "Add 'xgrammar', 'transformers' and 'accelerate' to requirements.pip.\n"
          f"[ComfyUI-MeshScript] import error: {e!r}\n"
          + traceback.format_exc())


if _XGR_AVAILABLE:
    class _TorchNativeXGRProcessor(_XGRLogitsProcessor):
        """xgrammar's HF LogitsProcessor, but applies the token bitmask via
        the 'torch_native' backend instead of the default 'triton' backend.

        The default backend requires Triton, which has no Windows wheel and
        is an extra (large) dependency on Linux. 'torch_native' is portable
        and, since the bitmask itself is precomputed by xgrammar's compiled
        grammar matcher, applying it is a cheap vectorized op regardless —
        measured at near-zero overhead vs. unconstrained generation.
        """

        def __call__(self, input_ids, scores):
            if len(self.matchers) == 0:
                self.batch_size = input_ids.shape[0]
                self.compiled_grammars = (
                    self.compiled_grammars
                    if len(self.compiled_grammars) > 1
                    else self.compiled_grammars * self.batch_size
                )
                self.matchers = [
                    xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)
                ]
                self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)

            if not self.prefilled:
                self.prefilled = True
            else:
                for i in range(self.batch_size):
                    if not self.matchers[i].is_terminated():
                        sampled_token = input_ids[i][-1].item()
                        assert self.matchers[i].accept_token(sampled_token)

            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

            xgr.apply_token_bitmask_inplace(
                scores, self.token_bitmask.to(scores.device), backend="torch_native"
            )
            return scores


    # Compiling a grammar from a JSON Schema takes ~0.1s — cheap, but the
    # schema is identical across every generate()/retry call for a given
    # node, so cache the compiled grammar per (tokenizer, schema).
    _grammar_cache: dict = {}

    def _compiled_grammar(model, tokenizer, schema: dict):
        key = (id(tokenizer), json.dumps(schema, sort_keys=True))
        cg = _grammar_cache.get(key)
        if cg is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=model.config.vocab_size
            )
            cg = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(schema)
            _grammar_cache[key] = cg
        return cg


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

    compiled_grammar = _compiled_grammar(model, tokenizer, schema)
    xgr_processor = _TorchNativeXGRProcessor(compiled_grammar)

    print(f"[_constrained_generate] prompt_tokens={inputs.input_ids.shape[1]}  "
          f"max_new_tokens={max_tokens}  temperature={temperature}")

    gen_kwargs = dict(
        max_new_tokens   = max_tokens,
        pad_token_id     = tokenizer.eos_token_id,
        eos_token_id     = tokenizer.eos_token_id,
        logits_processor = [xgr_processor],
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
        if not _XGR_AVAILABLE:
            raise RuntimeError(
                "xgrammar is not installed.\n"
                "Add 'xgrammar', 'transformers' and 'accelerate' to requirements.pip.\n"
                f"Import error was: {_XGR_IMPORT_ERROR!r}"
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
        if not _XGR_AVAILABLE:
            raise RuntimeError(
                "xgrammar is not installed.\n"
                "Add 'xgrammar', 'transformers' and 'accelerate' to requirements.pip.\n"
                f"Import error was: {_XGR_IMPORT_ERROR!r}"
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
