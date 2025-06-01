
import time
import psutil
import inspect
import functools
import gc
from types import FrameType
from typing import Dict

try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:           # CPU-only environment
    _GPU_AVAILABLE = False



GB = 1024 ** 3

####
#Memory Reporting
def _gpu_mem() -> Dict[str, float]:
    if not _GPU_AVAILABLE:
        return {"gpu_used_GB": 0.0, "gpu_free_GB": 0.0}
    free, total = cp.cuda.runtime.memGetInfo()
    used = total - free
    return {
        "gpu_used_GB": round(used / GB, 3),
        "gpu_free_GB": round(free / GB, 3),
    }


def _scan_locals(frame: FrameType, top: int = 10) -> Dict[str, str]:
    """Return the *top* largest numpy / cupy arrays in the given frame."""
    sizes = []
    for name, val in frame.f_locals.items():
        if hasattr(val, "nbytes"):
            sizes.append((val.nbytes, name, val.shape if hasattr(val, "shape") else None))
    sizes.sort(reverse=True)
    out = {n: f"{round(sz / GB, 3)} GB {sh}" for sz, n, sh in sizes[:top]}
    return out


def _emit(msg: str, logger):
    (logger.info if logger else print)(msg)

def profile_mem(logger=None, *, top_vars: int = 10):
    proc = psutil.Process()

    def _discover_logger(args, kwargs, func):
        if logger is not None:  # explicit beats implicit
            return logger
        if "logger" in kwargs and hasattr(kwargs["logger"], "info"):
            return kwargs["logger"]
        sig = inspect.signature(func).parameters
        if "logger" in sig:
            pos = list(sig).index("logger")
            if pos < len(args) and hasattr(args[pos], "info"):
                return args[pos]
        return None  # fallback → print

    def _find_settings(args, kwargs, func):
        """Return the settings object passed to *func*, or None."""
        # keyword case first
        if "settings" in kwargs:
            return kwargs["settings"]

        # positional: locate index of the 'settings' param in the signature
        sig = inspect.signature(func).parameters
        if "settings" in sig:
            pos = list(sig).index("settings")
            if pos < len(args):
                return args[pos]
        return None

    def _want_check(args, kwargs, func):
        s = _find_settings(args, kwargs, func)
        if s is not None and hasattr(s, "checkmem"):
            return bool(s.checkmem)
        return False  # default – profile

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            if not _want_check(args, kwargs, func):
                return func(*args, **kwargs)

            log = _discover_logger(args, kwargs, func)
            ram_pre = proc.memory_info().rss
            gpu_pre = _gpu_mem()
            t0 = time.perf_counter()
            res = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            ram_post = proc.memory_info().rss
            gpu_post = _gpu_mem()

            _emit(
                f"     [{func.__name__}] ΔRAM {round((ram_post-ram_pre)/GB,3)} GB "
                f"     ΔGPU {round(gpu_post['gpu_used_GB']-gpu_pre['gpu_used_GB'],3)} GB "
                f"\n     RAM {round(ram_post/GB,3)} GB "
                f"     GPU {gpu_post['gpu_used_GB']} GB "
                f"     {dt:.2f}s", log)

            frame = inspect.currentframe().f_back
            if frame:
                big = _scan_locals(frame, top_vars)
                if big:
                    _emit(f"     [{func.__name__}] top{top_vars}: {big}", log)

            gc.collect()
            return res
        return wrapper
    return decorator

