from .nodes        import NODE_CLASS_MAPPINGS     as _M1, NODE_DISPLAY_NAME_MAPPINGS     as _D1
from .nodes_llm    import NODE_CLASS_MAPPINGS     as _M2, NODE_DISPLAY_NAME_MAPPINGS     as _D2
from .nodes_canvas import NODE_CLASS_MAPPINGS     as _M3, NODE_DISPLAY_NAME_MAPPINGS     as _D3
from .nodes_ir     import NODE_CLASS_MAPPINGS     as _M4, NODE_DISPLAY_NAME_MAPPINGS     as _D4

NODE_CLASS_MAPPINGS        = {**_M1, **_M2, **_M3, **_M4}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2, **_D3, **_D4}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
