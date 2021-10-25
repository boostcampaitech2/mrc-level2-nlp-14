from .readers import (
    ExtractiveReader,
    AbstractiveReader,
    EnsembleReader,
    RetroReader,
)

READER_HOST = {
    "extractive": ExtractiveReader,
    "abstractive": AbstractiveReader,
    "ensemble": EnsembleReader,
    "retro": RetroReader,
}