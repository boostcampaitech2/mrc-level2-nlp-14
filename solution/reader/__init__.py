from .readers import (
    ExtractiveReader,
    GenerativeReader,
    EnsembleReader,
    RetroReader,
)

READER_HOST = {
    "extractive": ExtractiveReader,
    "generative": GenerativeReader,
    "ensemble": EnsembleReader,
    "retro": RetroReader,
}