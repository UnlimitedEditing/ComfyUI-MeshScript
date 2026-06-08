"""
ComfyUI-MeshScript — nodes.py

Foundation nodes:

    MeshScriptExecute       Run a MeshScript, get back TRIMESH + IMAGE renders
    SaveMeshWithScript      Export GLB with source script embedded in extras
    LoadMeshWithScript      Load GLB + extract embedded script

TRIMESH type is the same raw trimesh.Trimesh object used by ComfyUI-TripoSG,
so these nodes wire directly into the existing TripoSG pipeline.

Meshscript library resolution order
-------------------------------------
1. MESHSCRIPT_PATH env var (explicit override)
2. ../meshscript/  (sibling custom_nodes dir — Graydient deployment)
3. D:\\tripostl\\meshscript  (local dev machine)
"""

import os
import sys
import struct
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import folder_paths

# ── locate meshscript library ─────────────────────────────────────────────────

def _find_meshscript_root() -> str | None:
    def _valid(p):
        return os.path.isdir(os.path.join(p, "sandbox")) and \
               os.path.isdir(os.path.join(p, "ops"))

    if os.environ.get("MESHSCRIPT_PATH"):
        p = os.environ["MESHSCRIPT_PATH"]
        if _valid(p):
            return p

    # Sibling custom_nodes/meshscript — Graydient / standard deployment
    here   = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    for candidate in ["meshscript", "MeshScript"]:
        p = os.path.join(parent, candidate)
        if _valid(p):
            return p

    # Local dev path
    local = r"D:\tripostl\meshscript"
    if _valid(local):
        return local

    return None


_MS_ROOT = _find_meshscript_root()
if _MS_ROOT and _MS_ROOT not in sys.path:
    sys.path.insert(0, _MS_ROOT)
    print(f"[ComfyUI-MeshScript] meshscript library: {_MS_ROOT}")
else:
    print("[ComfyUI-MeshScript] WARNING — meshscript library not found. "
          "Set MESHSCRIPT_PATH env var or place meshscript/ as a sibling custom_nodes directory.")


# ── GLB extras (inline copy so node pack has no external dep on glb_extras.py) ─

_MAGIC      = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN  = 0x004E4942


def _parse_glb_chunks(data: bytes) -> list:
    magic, _, _ = struct.unpack_from("<III", data, 0)
    assert magic == _MAGIC, "Not a GLB file"
    chunks, off = [], 12
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunks.append((ctype, data[off + 8: off + 8 + clen]))
        off += 8 + clen
    return chunks


def _pack_glb(chunks: list) -> bytes:
    body = b"".join(struct.pack("<II", len(d), t) + d for t, d in chunks)
    return struct.pack("<III", _MAGIC, 2, 12 + len(body)) + body


def _pad4(data: bytes, pad_byte: bytes) -> bytes:
    return data + pad_byte * ((4 - len(data) % 4) % 4)


def _glb_inject(glb_bytes: bytes, script: str, spec: str) -> bytes:
    chunks = _parse_glb_chunks(glb_bytes)
    out = []
    for ctype, cdata in chunks:
        if ctype == _CHUNK_JSON:
            gltf = json.loads(cdata.rstrip(b"\x00 "))
            ex   = gltf.setdefault("extras", {})
            ex["meshscript_source"] = script
            ex["meshscript_spec"]   = spec
            raw = json.dumps(gltf, separators=(",", ":")).encode()
            out.append((_CHUNK_JSON, _pad4(raw, b" ")))
        else:
            out.append((ctype, cdata))
    return _pack_glb(out)


def _glb_extract(glb_bytes: bytes):
    try:
        for ctype, cdata in _parse_glb_chunks(glb_bytes):
            if ctype == _CHUNK_JSON:
                ex = json.loads(cdata.rstrip(b"\x00 ")).get("extras", {})
                return ex.get("meshscript_source", ""), ex.get("meshscript_spec", "")
    except Exception:
        pass
    return "", ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _renders_to_tensor(renders: list) -> torch.Tensor:
    """
    Convert list of {"image": np.ndarray (H,W,3) uint8} to
    ComfyUI IMAGE tensor (N, H, W, 3) float32 [0,1].
    Returns a 1×64×64×3 black tensor if no renders.
    """
    if not renders:
        return torch.zeros(1, 64, 64, 3)
    imgs = np.stack([r["image"][:, :, :3] for r in renders])  # (N,H,W,3) uint8
    return torch.from_numpy(imgs.astype(np.float32) / 255.0)


def _load_trimesh_from_bytes(glb_bytes: bytes):
    import trimesh as tm
    scene = tm.load(BytesIO(glb_bytes), file_type="glb", force="scene")
    if hasattr(scene, "geometry") and scene.geometry:
        return tm.util.concatenate(list(scene.geometry.values()))
    return scene  # already a Trimesh


def _fetch_bytes(url_or_path: str) -> bytes:
    """Download URL or read local file, return raw bytes."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        import urllib.request, ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        with urllib.request.urlopen(url_or_path, timeout=60, context=ctx) as r:
            return r.read()
    with open(url_or_path, "rb") as f:
        return f.read()


# ── Node: MeshScriptExecute ───────────────────────────────────────────────────

class MeshScriptExecute:
    """
    Run a MeshScript program and return the final mesh + multi-view renders.

    render_views = 0  →  skip rendering (fast, no IMAGE output)
    render_views > 0  →  render N evenly-spaced azimuth views at render_size px
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True,
                    "default":   "# paste or connect MeshScript here\nresult = box(1, 1, 1)\nshow(result, 'result')",
                }),
                "render_views": ("INT", {
                    "default": 4, "min": 0, "max": 8,
                    "tooltip": "Number of render views (0 = skip renders)",
                }),
                "render_size": ("INT", {
                    "default": 256, "min": 64, "max": 1024, "step": 64,
                    "tooltip": "Square render resolution in pixels",
                }),
            },
            "optional": {
                "spec": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("TRIMESH", "IMAGE",   "STRING", "STRING",  "BOOLEAN")
    RETURN_NAMES  = ("trimesh", "renders", "script", "error",   "success")
    FUNCTION      = "execute"
    CATEGORY      = "MeshScript"

    def execute(self, script: str, render_views: int, render_size: int, spec: str = ""):
        if _MS_ROOT is None:
            err = "meshscript library not found — set MESHSCRIPT_PATH"
            return (None, torch.zeros(1, 64, 64, 3), script, err, False)

        from sandbox.executor import run as ms_run

        render_config = None
        if render_views > 0:
            render_config = {
                "views":  render_views,
                "width":  render_size,
                "height": render_size,
            }

        result = ms_run(script, reference=spec, render_config=render_config, export_dir=None)

        # Use the last checkpoint mesh even on partial failure — if any show()
        # calls ran we have valid geometry worth returning.  The error string
        # travels on the error output pin so the caller can decide what to do.
        mesh    = result.get("final")   # checkpoints[-1].mesh, or None
        renders = (result["checkpoints"][-1].get("renders", [])
                   if result["checkpoints"] else [])
        img_t   = _renders_to_tensor(renders)
        err     = result.get("error") or ""
        success = bool(result.get("success"))

        if mesh is None:
            return (None, img_t, script,
                    err or "script produced no geometry (no show() calls)", False)

        return (mesh, img_t, script, err, success)


# ── Node: SaveMeshWithScript ──────────────────────────────────────────────────

_FORMATS = ["glb", "obj", "ply", "stl", "3mf", "dae"]


class SaveMeshWithScript:
    """
    Export a TRIMESH to GLB with the generating MeshScript embedded in
    the file's GLTF extras.  The script travels with the mesh so that
    LoadMeshWithScript (or mesh→mesh workflow) can reconstruct it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh":         ("TRIMESH",),
                "script":          ("STRING",  {"default": ""}),
                "filename_prefix": ("STRING",  {"default": "MeshScript/mesh"}),
                "file_format":     (_FORMATS,),
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
    CATEGORY      = "MeshScript"
    OUTPUT_NODE   = True

    def save(self, trimesh, script: str, filename_prefix: str,
             file_format: str, spec: str = "", save_file: bool = True):
        save_dir = (folder_paths.get_output_directory() if save_file
                    else folder_paths.get_temp_directory())
        full_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, save_dir
        )
        Path(full_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(full_dir) / f"{fname}_{counter:05}_.{file_format}"

        if file_format == "glb":
            # Export to memory, inject script, write
            buf = BytesIO()
            trimesh.export(buf, file_type="glb")
            glb_bytes = _glb_inject(buf.getvalue(), script, spec)
            out_path.write_bytes(glb_bytes)
        else:
            trimesh.export(str(out_path), file_type=file_format)

        rel_path = str(Path(subfolder) / f"{fname}_{counter:05}_.{file_format}")
        print(f"[SaveMeshWithScript] {out_path}  ({out_path.stat().st_size} bytes)")
        return (rel_path,)


# ── Node: LoadMeshWithScript ──────────────────────────────────────────────────

class LoadMeshWithScript:
    """
    Load a GLB file (from URL or local path) and extract both the mesh geometry
    and the embedded MeshScript source code.

    Use the script output as input to MeshScriptLLMRevise for mesh→mesh editing.
    If no script is embedded, script and spec are returned as empty strings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {
                    "default": "https://...",
                    "tooltip": "URL or absolute local path to a GLB file",
                }),
            },
        }

    RETURN_TYPES  = ("TRIMESH", "STRING",  "STRING")
    RETURN_NAMES  = ("trimesh", "script",  "spec")
    FUNCTION      = "load"
    CATEGORY      = "MeshScript"

    def load(self, url_or_path: str):
        glb_bytes      = _fetch_bytes(url_or_path.strip())
        script, spec   = _glb_extract(glb_bytes)
        mesh           = _load_trimesh_from_bytes(glb_bytes)
        return (mesh, script, spec)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MeshScriptExecute":       MeshScriptExecute,
    "SaveMeshWithScript":      SaveMeshWithScript,
    "LoadMeshWithScript":      LoadMeshWithScript,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshScriptExecute":       "MeshScript Execute",
    "SaveMeshWithScript":      "Save Mesh With Script",
    "LoadMeshWithScript":      "Load Mesh With Script",
}
