from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

import numpy as np

from . import Model, Session, feed_batch

log = logging.getLogger("ears")

THRESHOLD_NAMES = ("ep_thr", "ep_deb", "barge_thr", "barge_deb", "bc_thr", "bc_deb")
_INT = {"ep_deb", "barge_deb", "bc_deb"}


class EarsSession:
    """one conversation, thresholds read and write through to the native session"""

    def __init__(self, sess: Session):
        object.__setattr__(self, "_s", sess)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name):
        if name in THRESHOLD_NAMES:
            return self._s.params[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in THRESHOLD_NAMES:
            self._s.set_params(**{name: int(value) if name in _INT else float(value)})
        else:
            object.__setattr__(self, name, value)

    def reset(self):
        self._s.reset()


class TygerEars:
    """drop in for the reference ears plugin: open, close, async feed of one hop, batched across conversations"""

    THRESHOLDS = THRESHOLD_NAMES

    def __init__(self, model: str | Model, slots: int = 4, gather_ms: float = 15, expect_ms: float = 60,
                 encoder: str = "exact", backend: str = "auto", asr: bool = True, n_threads: int = 0, **defaults):
        self.model = model if isinstance(model, Model) else Model(model, backend=backend, asr=asr, n_threads=n_threads)
        self.sr = self.model.sample_rate
        self.hop_samples = self.model.hop_samples
        self.slots = slots
        self.gather_s = gather_ms / 1e3
        self.expect_s = expect_ms / 1e3
        self.encoder = encoder
        base = self.model.default_params()
        self.defaults = {k: base[k] for k in THRESHOLD_NAMES}
        self.defaults.update({k: v for k, v in defaults.items() if k in THRESHOLD_NAMES})
        self._extra = {k: v for k, v in defaults.items() if k not in THRESHOLD_NAMES}
        self._lock = threading.Lock()
        self._sessions: set[EarsSession] = set()
        self._expect: set[EarsSession] = set()
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, name="tyger-ears", daemon=True).start()

    def open(self, **thresholds) -> EarsSession | None:
        return self.open_session({**self.defaults, **thresholds})

    def _live(self) -> int:
        return sum(not s.closed for s in self._sessions)

    def open_session(self, thresholds: dict) -> EarsSession | None:
        with self._lock:
            if self._live() >= self.slots:
                return None
            s = EarsSession(self.model.session(encoder=self.encoder, **self._extra, **thresholds))
            self._sessions.add(s)
            return s

    def thresholds(self, session: EarsSession) -> dict:
        p = session._s.params
        return {k: p[k] for k in THRESHOLD_NAMES}

    def set_threshold(self, session: EarsSession, name: str, value):
        if name not in THRESHOLD_NAMES:
            raise KeyError(name)
        setattr(session, name, value)
        return getattr(session, name)

    def close(self, session: EarsSession):
        session.closed = True
        self._q.put(("close", session))

    def full(self) -> bool:
        with self._lock:
            return self._live() >= self.slots

    def active(self) -> int:
        with self._lock:
            return self._live()

    async def feed(self, session: EarsSession, user_hop, agent_hop, mode, more=False) -> list[dict]:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._q.put(("feed", session, user_hop, agent_hop, mode, fut, loop, more))
        return await fut

    @staticmethod
    def _resolve(item, events=None, exc=None):
        fut, loop = item[5], item[6]

        def _set():
            if not fut.done():
                fut.set_exception(exc) if exc is not None else fut.set_result(events)
        loop.call_soon_threadsafe(_set)

    def _loop(self):
        carry = []
        while True:
            items = carry or [self._q.get()]
            carry = []
            t0 = time.perf_counter()
            while True:
                with self._lock:
                    n_live = self._live()
                have = {it[1] for it in items if it[0] == "feed"}
                if len(have) >= n_live:
                    break
                waiting = any(s not in have and not s.closed for s in self._expect)
                rem = t0 + (self.expect_s if waiting else self.gather_s) - time.perf_counter()
                if rem <= 0:
                    break
                try:
                    items.append(self._q.get(timeout=rem))
                except queue.Empty:
                    break
            feeds, seen = [], set()
            for it in items:
                if it[0] != "feed":
                    continue
                if it[1].closed:
                    self._resolve(it, [])
                elif it[1] in seen:
                    carry.append(it)  # second hop of the same conversation goes in the next tick
                else:
                    seen.add(it[1])
                    feeds.append(it)
            self._expect = {it[1] for it in feeds if it[7]}
            if feeds:
                try:
                    out = feed_batch(self.model, [it[1]._s for it in feeds], [it[2] for it in feeds],
                                     [None if it[3] is None else np.asarray(it[3]) for it in feeds],
                                     [it[4] for it in feeds])
                    for it, ev in zip(feeds, out):
                        self._resolve(it, ev)
                except Exception as e:
                    log.exception("tyger tick failed")
                    for it in feeds:
                        self._resolve(it, exc=e)
            for it in items:
                if it[0] == "close":
                    with self._lock:
                        self._sessions.discard(it[1])
                    it[1]._s.close()
