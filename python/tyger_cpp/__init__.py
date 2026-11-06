from __future__ import annotations

import ctypes as C
import json
import os
import sys
from pathlib import Path

import numpy as np

__all__ = ["Model", "Session", "feed_batch", "load_library", "SESSION_KEYS"]

BACKENDS = {"auto": 0, "cpu": 1, "vulkan": 2}
ENCODERS = {"exact": 0, "fast": 1}
MODES = {"listen": 0, "speaking": 1}


class _ModelParams(C.Structure):
    _fields_ = [("backend", C.c_int), ("device", C.c_int), ("n_threads", C.c_int), ("enable_asr", C.c_bool),
                ("verbose", C.c_bool)]


class _SessionParams(C.Structure):
    _fields_ = [("ep_thr", C.c_float), ("ep_deb", C.c_int), ("barge_thr", C.c_float), ("barge_deb", C.c_int),
                ("bc_thr", C.c_float), ("bc_deb", C.c_int), ("bc_refractory", C.c_int), ("theta", C.c_float),
                ("partial_refresh_s", C.c_float), ("fallback_sil_frames", C.c_int), ("encoder_mode", C.c_int),
                ("emit_frames", C.c_bool), ("prompt_id", C.c_int)]


SESSION_KEYS = tuple(n for n, _ in _SessionParams._fields_)

_lib = None


def _candidates():
    env = os.environ.get("TYGER_LIB")
    if env:
        yield Path(env)
    names = ["tyger.dll"] if sys.platform == "win32" else ["libtyger.dylib"] if sys.platform == "darwin" else ["libtyger.so"]
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    for d in [here, root / "build" / "bin" / "Release", root / "build" / "bin", root / "build"]:
        for n in names:
            yield d / n


def load_library(path: str | os.PathLike | None = None):
    """the shared lib, from path, TYGER_LIB, next to this package or the repo build folder"""
    global _lib
    if _lib is not None and path is None:
        return _lib
    tried = []
    for p in ([Path(path)] if path else _candidates()):
        tried.append(str(p))
        if p.is_file():
            lib = C.CDLL(str(p))
            break
    else:
        raise OSError("tyger shared library not found, tried: " + ", ".join(tried))

    vp = C.c_void_p
    lib.tyger_last_error.restype = vp
    lib.tyger_version.restype = vp
    lib.tyger_model_default_params.restype = _ModelParams
    lib.tyger_model_load.argtypes = [C.c_char_p, _ModelParams]
    lib.tyger_model_load.restype = vp
    lib.tyger_model_free.argtypes = [vp]
    lib.tyger_model_info.argtypes = [vp]
    lib.tyger_model_info.restype = vp
    lib.tyger_sample_rate.argtypes = [vp]
    lib.tyger_hop_samples.argtypes = [vp]
    lib.tyger_session_default_params.argtypes = [vp]
    lib.tyger_session_default_params.restype = _SessionParams
    lib.tyger_session_open.argtypes = [vp, C.POINTER(_SessionParams)]
    lib.tyger_session_open.restype = vp
    lib.tyger_session_close.argtypes = [vp]
    lib.tyger_session_reset.argtypes = [vp]
    lib.tyger_session_reset_keep.argtypes = [vp]
    lib.tyger_session_set_params.argtypes = [vp, C.POINTER(_SessionParams)]
    lib.tyger_session_get_params.argtypes = [vp, C.POINTER(_SessionParams)]
    fp = C.POINTER(C.c_float)
    lib.tyger_session_feed.argtypes = [vp, fp, fp, C.c_int, C.c_int, vp]
    lib.tyger_feed_batch.argtypes = [vp, C.c_int, C.POINTER(vp), C.POINTER(fp), C.POINTER(fp), C.POINTER(C.c_int)]
    lib.tyger_session_events.argtypes = [vp, vp]
    lib.tyger_session_events_json.argtypes = [vp]
    lib.tyger_session_events_json.restype = vp
    lib.tyger_transcribe.argtypes = [vp, fp, C.c_int, C.c_int]
    lib.tyger_transcribe.restype = vp
    lib.tyger_free.argtypes = [vp]
    lib.tyger_encode.argtypes = [vp, fp, C.c_int, C.c_int, C.c_int, fp, C.c_int]
    if path is None:
        _lib = lib
    return lib


def _str(p) -> str | None:
    return C.string_at(p).decode("utf-8", "replace") if p else None


def _err(lib) -> str:
    return _str(lib.tyger_last_error()) or "unknown error"


def _pcm(x, n: int | None = None) -> np.ndarray:
    """float32 contiguous 16 kHz mono, int16 input is scaled"""
    a = np.asarray(x)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    if n is not None and a.size != n:
        raise ValueError(f"expected {n} samples, got {a.size}")
    return a


def _fptr(a: np.ndarray):
    return a.ctypes.data_as(C.POINTER(C.c_float))


def version() -> str:
    return _str(load_library().tyger_version())


class Model:
    """a loaded gguf. backend is auto, cpu or vulkan"""

    def __init__(self, path: str | os.PathLike, backend: str = "auto", device: int = -1, n_threads: int = 0,
                 asr: bool = True, verbose: bool = False, lib: str | os.PathLike | None = None):
        self._lib = load_library(lib)
        p = self._lib.tyger_model_default_params()
        p.backend = BACKENDS[backend]
        p.device = device
        p.n_threads = n_threads
        p.enable_asr = asr
        p.verbose = verbose
        self._h = self._lib.tyger_model_load(os.fsencode(os.fspath(path)), p)
        if not self._h:
            raise RuntimeError("tyger: " + _err(self._lib))
        self.info = json.loads(_str(self._lib.tyger_model_info(self._h)))
        self.sample_rate = self._lib.tyger_sample_rate(self._h)
        self.hop_samples = self._lib.tyger_hop_samples(self._h)

    def close(self):
        if self._h:
            self._lib.tyger_model_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def default_params(self) -> dict:
        p = self._lib.tyger_session_default_params(self._h)
        return {k: getattr(p, k) for k in SESSION_KEYS}

    def session(self, **params) -> "Session":
        return Session(self, **params)

    def transcribe(self, pcm, prompt_id: int = -1) -> str:
        a = _pcm(pcm)
        r = self._lib.tyger_transcribe(self._h, _fptr(a), a.size, prompt_id)
        if not r:
            raise RuntimeError("tyger: " + _err(self._lib))
        try:
            return _str(r)
        finally:
            self._lib.tyger_free(r)

    def encode(self, pcm, layers: int = 15, lookahead: int = 0) -> np.ndarray:
        """debug, offline encoder output [frames, 1024]"""
        a = _pcm(pcm)
        cap = a.size // 1280 + 4
        out = np.zeros((cap, 1024), np.float32)
        n = self._lib.tyger_encode(self._h, _fptr(a), a.size, layers, lookahead, _fptr(out), cap)
        if n < 0:
            raise RuntimeError("tyger: " + _err(self._lib))
        return out[:n]


def _params_from(base: _SessionParams, kw: dict) -> _SessionParams:
    p = _SessionParams.from_buffer_copy(base)
    for k, v in kw.items():
        if k == "encoder":
            k, v = "encoder_mode", ENCODERS[v] if isinstance(v, str) else int(v)
        elif k == "frames":
            k = "emit_frames"
        if k not in SESSION_KEYS:
            raise KeyError(f"unknown session param {k}")
        setattr(p, k, v)
    return p


class Session:
    """one conversation. feed() takes any number of samples and returns the events of every hop it completed"""

    def __init__(self, model: Model, **params):
        self.model = model
        self._lib = model._lib
        p = _params_from(self._lib.tyger_session_default_params(model._h), params)
        self._h = self._lib.tyger_session_open(model._h, C.byref(p))
        if not self._h:
            raise RuntimeError("tyger: " + _err(self._lib))

    @property
    def params(self) -> dict:
        p = _SessionParams()
        self._lib.tyger_session_get_params(self._h, C.byref(p))
        return {k: getattr(p, k) for k in SESSION_KEYS}

    def set_params(self, **kw):
        cur = _SessionParams()
        self._lib.tyger_session_get_params(self._h, C.byref(cur))
        p = _params_from(cur, kw)
        if self._lib.tyger_session_set_params(self._h, C.byref(p)) != 0:
            raise ValueError("tyger: " + _err(self._lib))

    def reset(self, keep: bool = False):
        """start a new turn, applied before the next hop. keep seeds it with the last 2 s of speaking mode user audio"""
        (self._lib.tyger_session_reset_keep if keep else self._lib.tyger_session_reset)(self._h)

    def feed(self, user, agent=None, mode: str | int = "listen") -> list[dict]:
        u = _pcm(user)
        a = None if agent is None else _pcm(agent, u.size)
        m = MODES[mode] if isinstance(mode, str) else int(mode)
        n = self._lib.tyger_session_feed(self._h, _fptr(u), None if a is None else _fptr(a), u.size, m, None)
        if n < 0:
            raise RuntimeError("tyger: " + _err(self._lib))
        return self.events() if n else []

    def events(self) -> list[dict]:
        """the last feed's events"""
        return json.loads(_str(self._lib.tyger_session_events_json(self._h)))

    def close(self):
        if self._h:
            self._lib.tyger_session_close(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def feed_batch(model: Model, sessions, users, agents=None, modes=None) -> list[list[dict]]:
    """exactly one hop for each session in one batched encode, returns each session's events"""
    n = len(sessions)
    if n == 0:
        return []
    hop = model.hop_samples
    us = [_pcm(u, hop) for u in users]
    ag = [None if agents is None or agents[i] is None else _pcm(agents[i], hop) for i in range(n)]
    zeros = np.zeros(hop, np.float32)
    ag = [zeros if a is None else a for a in ag]
    fp = C.POINTER(C.c_float)
    hs = (C.c_void_p * n)(*[s._h for s in sessions])
    up = (fp * n)(*[_fptr(u) for u in us])
    apt = (fp * n)(*[_fptr(a) for a in ag])
    ms = (C.c_int * n)(*[MODES[m] if isinstance(m, str) else int(m) for m in (modes or ["listen"] * n)])
    if model._lib.tyger_feed_batch(model._h, n, hs, up, apt, ms) != 0:
        raise RuntimeError("tyger: " + _err(model._lib))
    out = []
    for s in sessions:
        k = model._lib.tyger_session_events(s._h, None)
        out.append(s.events() if k > 0 else [])
    return out
