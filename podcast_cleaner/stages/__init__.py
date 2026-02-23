"""Pipeline stage modules."""

STAGE_ORDER = [
    "download",
    "preprocess",
    "separate",
    "denoise",
    "transcribe",
    "normalize",
    "export",
]
