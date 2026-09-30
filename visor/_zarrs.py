"""Shared zarrs acceleration setup.

Selecting the zarrs codec pipeline is a process-global zarr setting; apply it
idempotently on demand (constructors call this) instead of at import time.
Disable by setting the environment variable VISOR_PY_DISABLE_ZARRS=1.
"""
import os

import zarr
import zarrs  # noqa: F401  (import registers the codec pipeline path)

_applied = False


def enable_zarrs_acceleration() -> bool:
    """Idempotently select the zarrs codec pipeline for zarr I/O.

    Returns True when the pipeline is (already) active, False when disabled
    via the VISOR_PY_DISABLE_ZARRS environment variable.
    """
    global _applied
    if os.environ.get('VISOR_PY_DISABLE_ZARRS', '') not in ('', '0', 'false', 'False'):
        return False
    if not _applied:
        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
        _applied = True
    return True
