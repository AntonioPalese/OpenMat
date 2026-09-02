"""
CUDA streams.

Streams are the canonical execution path in OpenMat: the synchronous API is a
thin delegate over the stream API with the default (null) stream.  In Python:

    s = Stream()
    c = a.add(b, stream=s)      # enqueued, not finished
    s.synchronize()             # now c is safe to read
    print(c.sum())

    with Stream() as s:         # synchronizes on exit
        c = a.add(b, stream=s)

Lifetime: memory for a tensor produced on a stream is allocated from that
stream's pool and must be freed on the same stream, so every result tensor
keeps a reference to the Stream that produced it.  Do not close a Stream while
tensors made on it are still alive.
"""
from ._clib import CLIB, _errbuf, _check_ptr, _check_int, _ERR_LEN


class Stream:
    """Owning wrapper around a cudaStream_t (om::Stream on the C++ side).

    The handle is reference-counted in the C layer, and every tensor allocated
    on the stream holds one of those references.  That keeps the stream alive
    until the last tensor freed on it is gone, whatever order Python decides to
    finalize objects in — see the StreamBox comment in openmat_capi.cpp.
    """

    __slots__ = ("_h", "__weakref__")

    def __init__(self, _handle=None, _owns=True):
        if _handle is None and _owns:
            eb = _errbuf()
            _handle = _check_ptr(CLIB.om_stream_create(eb, _ERR_LEN), eb)
        self._h = _handle

    @classmethod
    def _from_handle(cls, handle) -> "Stream":
        """A second facade over an existing stream; takes its own reference."""
        if not handle:
            return _DEFAULT
        CLIB.om_stream_retain(handle)
        s = cls.__new__(cls)
        s._h = handle
        return s

    @staticmethod
    def default() -> "Stream":
        """Non-owning wrapper around the default (null) stream."""
        return _DEFAULT

    @property
    def is_default(self) -> bool:
        return self._h is None

    @property
    def handle(self) -> int:
        """The raw cudaStream_t as an integer (0 for the default stream)."""
        return CLIB.om_stream_handle(self._h) or 0

    def synchronize(self) -> "Stream":
        """Block until every operation enqueued on this stream has completed."""
        eb = _errbuf()
        _check_int(CLIB.om_stream_synchronize(self._h, eb, _ERR_LEN), eb)
        return self

    def close(self):
        """Drop this object's reference to the stream. Idempotent.

        The underlying cudaStream_t is destroyed once the last reference goes —
        including the ones held by tensors whose memory came from its pool and
        must be freed on it.  Closing early is therefore safe rather than a
        use-after-free; those tensors go on working.
        """
        h, self._h = self._h, None
        if h:
            CLIB.om_stream_release(h)

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc):
        # Synchronize even on the error path: work is already in flight and the
        # tensors it writes into may be freed as the exception unwinds.
        self.synchronize()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return "Stream(default)" if self.is_default else f"Stream(handle=0x{self.handle:x})"


_DEFAULT = Stream(_handle=None, _owns=False)


def _handle_of(stream):
    """None or a Stream -> the void* to hand to the C API (None = default stream)."""
    if stream is None:
        return None
    if not isinstance(stream, Stream):
        raise TypeError(f"stream must be an openmat.Stream, got {type(stream).__name__}")
    return stream._h


def synchronize():
    """Block until every stream on the current device has completed (cudaDeviceSynchronize)."""
    eb = _errbuf()
    _check_int(CLIB.om_device_synchronize(eb, _ERR_LEN), eb)


def cuda_is_available() -> bool:
    return bool(CLIB.om_cuda_is_available())


def device_count() -> int:
    return int(CLIB.om_cuda_device_count())
