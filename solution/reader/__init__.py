from .readers import (
    ExtractiveReader,
    GenerativeReader,
    EnsembleReader,
)

READER_HOST = {
    "extractive": ExtractiveReader,
    "generative": GenerativeReader,
    "ensemble": EnsembleReader,
}
