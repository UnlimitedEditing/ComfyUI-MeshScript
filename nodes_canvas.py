"""
ComfyUI-MeshScript — nodes_canvas.py

CanvasScript nodes — the 2D sibling of MeshScriptExecute:

    CanvasScriptExecute     Run a CanvasScript, get back IMAGE
    SaveCanvasWithScript    Save IMAGE as PNG with source script embedded in metadata
    LoadCanvasWithScript    Load PNG + extract embedded script

Uses the same meshscript library resolution as nodes.py (_MS_ROOT);
canvas_ops/ and canvas_sandbox/ live alongside ops/ and sandbox/ at the
meshscript repo root, so no extra path setup is needed.
"""

import urllib.request
import ssl
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .nodes import _MS_ROOT, _renders_to_tensor


def _fetch_bytes(url_or_path: str) -> bytes:
    """Download URL or read local file, return raw bytes."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url_or_path, timeout=60, context=ctx) as r:
            return r.read()
    with open(url_or_path, "rb") as f:
        return f.read()


_DEFAULT_SCRIPT = (
    "# paste or connect CanvasScript here\n"
    "doc = new_document(768, 768, background=(255, 255, 255, 255))\n"
    "title = text(\"Hello, CanvasScript\", size=64, color=(20, 20, 30, 255))\n"
    "doc = align(doc, title, 'center')\n"
    "show(doc, 'result')\n"
)


# ── Node: CanvasScriptExecute ───────────────────────────────────────────────────

class CanvasScriptExecute:
    """
    Run a CanvasScript program and return the final composited canvas as IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True,
                    "default":   _DEFAULT_SCRIPT,
                }),
            },
            "optional": {
                "spec": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES  = ("image",  "script", "error",  "success")
    FUNCTION      = "execute"
    CATEGORY      = "CanvasScript"

    def execute(self, script: str, spec: str = ""):
        if _MS_ROOT is None:
            err = "meshscript library not found — set MESHSCRIPT_PATH"
            return (torch.zeros(1, 64, 64, 3), script, err, False)

        from canvas_sandbox.executor import run as cs_run

        print(f"[CanvasScriptExecute] spec={spec!r}")
        print(f"[CanvasScriptExecute] script ({len(script)} chars, "
              f"{script.count('show(')} show() calls):")
        for i, line in enumerate(script.splitlines(), 1):
            print(f"  {i:3d} | {line}")

        result = cs_run(script, reference=spec, export_dir=None)

        renders = (result["checkpoints"][-1].get("renders", [])
                   if result["checkpoints"] else [])
        img_t   = _renders_to_tensor(renders)
        err     = result.get("error") or ""
        success = bool(result.get("success"))

        print(f"[CanvasScriptExecute] success={success}  "
              f"checkpoints={len(result['checkpoints'])}")
        if err:
            print(f"[CanvasScriptExecute] execution error:\n{err}")

        if not renders:
            return (img_t, script,
                    err or "script produced no image (no show() calls)", False)

        return (img_t, script, err, success)


# ── Node: SaveCanvasWithScript ──────────────────────────────────────────────────

class SaveCanvasWithScript:
    """
    Save an IMAGE to PNG with the generating CanvasScript embedded as PNG
    text metadata.  The script travels with the file so that
    LoadCanvasWithScript (or image->image workflow) can reconstruct it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "script":          ("STRING",  {"default": ""}),
                "filename_prefix": ("STRING",  {"default": "CanvasScript/canvas"}),
            },
            "optional": {
                "spec":      ("STRING",  {"default": ""}),
                "save_file": ("BOOLEAN", {"default": True,
                                          "label_on": "output", "label_off": "temp"}),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("file_path",)
    FUNCTION      = "save"
    CATEGORY      = "CanvasScript"
    OUTPUT_NODE   = True

    def save(self, image, script: str, filename_prefix: str,
             spec: str = "", save_file: bool = True):
        save_dir = (folder_paths.get_output_directory() if save_file
                    else folder_paths.get_temp_directory())
        full_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, save_dir
        )
        Path(full_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(full_dir) / f"{fname}_{counter:05}_.png"

        arr = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

        meta = PngInfo()
        meta.add_text("canvasscript_source", script)
        meta.add_text("canvasscript_spec", spec)
        img.save(out_path, pnginfo=meta)

        rel_path = str(Path(subfolder) / f"{fname}_{counter:05}_.png")
        print(f"[SaveCanvasWithScript] {out_path}  ({out_path.stat().st_size} bytes)")
        return (rel_path,)


# ── Node: LoadCanvasWithScript ──────────────────────────────────────────────────

class LoadCanvasWithScript:
    """
    Load a PNG file (from URL or local path) and extract both the image and
    the embedded CanvasScript source code.

    Use the script output as input to a future CanvasScript revise node for
    image->image workflow. If no script is embedded, script and spec are
    returned as empty strings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {
                    "default": "https://...",
                    "tooltip": "URL or absolute local path to a PNG file",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES  = ("image",  "script", "spec")
    FUNCTION      = "load"
    CATEGORY      = "CanvasScript"

    def load(self, url_or_path: str):
        data = _fetch_bytes(url_or_path.strip())
        img  = Image.open(BytesIO(data))

        script = img.info.get("canvasscript_source", "")
        spec   = img.info.get("canvasscript_spec", "")

        rgb = img.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]

        return (tensor, script, spec)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "CanvasScriptExecute":     CanvasScriptExecute,
    "SaveCanvasWithScript":    SaveCanvasWithScript,
    "LoadCanvasWithScript":    LoadCanvasWithScript,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanvasScriptExecute":     "CanvasScript Execute",
    "SaveCanvasWithScript":    "Save Canvas With Script",
    "LoadCanvasWithScript":    "Load Canvas With Script",
}
