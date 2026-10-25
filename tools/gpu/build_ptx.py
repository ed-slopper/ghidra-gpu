"""Compile src/main/cuda/*.cu to PTX with NVRTC and write them as Java resources.

Why NVRTC and not nvcc: nvcc needs a supported host C++ compiler even for -ptx,
NVRTC needs nothing but its own DLL. `pip install nvidia-cuda-nvrtc-cu12` is
enough to regenerate the kernels on any machine.

The PTX is committed (src/main/resources/com/xebyte/gpu/) so building the Java
extension never needs CUDA; the driver JIT-compiles it for whatever GPU is
present on first load. compute_52 (Maxwell) is the floor so one PTX file runs on
every GPU CUDA 12 drivers support.

Usage:
    python -m tools.gpu.build_ptx              # rebuild
    python -m tools.gpu.build_ptx --check      # fail if the committed PTX is stale
    NVRTC_DLL=C:/path/nvrtc64_120_0.dll python -m tools.gpu.build_ptx
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import site
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "src" / "main" / "cuda"
OUT_DIR = REPO / "src" / "main" / "resources" / "com" / "xebyte" / "gpu"
ARCH = "compute_52"
OPTIONS = [f"--gpu-architecture={ARCH}", "--std=c++14", "-default-device", "--fmad=false"]


def find_nvrtc() -> str:
    env = os.environ.get("NVRTC_DLL")
    if env:
        return env
    names = ["nvrtc64_*.dll", "libnvrtc.so*"]
    roots = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        roots += [os.path.join(sp, "nvidia", "cuda_nvrtc", "bin"), os.path.join(sp, "nvidia", "cuda_nvrtc", "lib")]
    if os.environ.get("CUDA_PATH"):
        roots += [os.path.join(os.environ["CUDA_PATH"], "bin"), os.path.join(os.environ["CUDA_PATH"], "lib64")]
    roots += ["/usr/local/cuda/lib64"]
    for root in roots:
        for pat in names:
            hits = sorted(h for h in glob.glob(os.path.join(root, pat)) if "builtins" not in h and ".alt." not in h)
            if hits:
                return hits[-1]
    raise SystemExit("NVRTC not found: pip install nvidia-cuda-nvrtc-cu12, or set NVRTC_DLL")


def compile_ptx(nvrtc, source: str, name: str) -> str:
    prog = ctypes.c_void_p()
    check(nvrtc, nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None))
    opts = (ctypes.c_char_p * len(OPTIONS))(*[o.encode() for o in OPTIONS])
    rc = nvrtc.nvrtcCompileProgram(prog, len(OPTIONS), opts)
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise SystemExit(f"{name}: NVRTC compile failed\n{log.value.decode(errors='replace')}")
    if log.value.strip():
        print(log.value.decode(errors="replace"), file=sys.stderr)
    size = ctypes.c_size_t()
    check(nvrtc, nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)))
    buf = ctypes.create_string_buffer(size.value)
    check(nvrtc, nvrtc.nvrtcGetPTX(prog, buf))
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.value.decode()


def check(nvrtc, rc: int) -> None:
    if rc != 0:
        nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        raise SystemExit(f"NVRTC error {rc}: {nvrtc.nvrtcGetErrorString(rc).decode()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed PTX differs")
    args = ap.parse_args()
    dll = find_nvrtc()
    if os.name == "nt":
        # nvrtc loads nvrtc-builtins64_*.dll itself with the default search order,
        # which add_dll_directory does not affect -- PATH does.
        os.add_dll_directory(os.path.dirname(dll))
        os.environ["PATH"] = os.path.dirname(dll) + os.pathsep + os.environ.get("PATH", "")
    nvrtc = ctypes.CDLL(dll)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for cu in sorted(SRC_DIR.glob("*.cu")):
        ptx = compile_ptx(nvrtc, cu.read_text(encoding="utf-8"), cu.name)
        out = OUT_DIR / (cu.stem + ".ptx")
        if args.check:
            # compare without line endings: git may check the PTX out with CRLF on Windows
            def norm(t): return "\n".join(t.splitlines())
            if not out.exists() or norm(out.read_text(encoding="utf-8")) != norm(ptx):
                stale.append(out)
            continue
        out.write_text(ptx, encoding="utf-8", newline="\n")
        print(f"{cu.relative_to(REPO)} -> {out.relative_to(REPO)} ({len(ptx):,} chars, {ARCH}, {Path(dll).name})")
    if stale:
        print("stale PTX (run python -m tools.gpu.build_ptx):", *map(str, stale), sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
