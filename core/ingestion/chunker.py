"""
Adaptive chunking strategy.
Chunk sizes are determined by FILE EXTENSION AND FOLDER LOCATION only.
NEVER by domain-specific keywords (no "transcript", "pricing" etc. as content triggers).
This ensures the system works for ANY client industry.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkConfig:
    chunk_size: int      # tokens
    chunk_overlap: int


# Chunk sizes by source folder name (user-controlled Drive folder names)
# These are just sensible defaults — folder names in Drive are the user's choice
def get_chunk_config(file_path: str, source_folder: str) -> ChunkConfig:
    """
    Returns adaptive chunking config based on source folder.
    NO domain-specific keywords — folder name drives the logic.
    Larger overlap for transcripts because critical facts often span speaker turns.
    """
    configs = {
        # Transcripts: long conversational content, facts scattered across turns
        # High overlap (200) ensures context isn't lost at chunk boundaries
        "transcripts": ChunkConfig(chunk_size=512, chunk_overlap=200),
        
        # Pricing: precise numerical facts, keep chunks tight for accuracy
        "pricing":     ChunkConfig(chunk_size=256, chunk_overlap=50),
        
        # Brand voice: style rules, medium overlap
        "brand_voice": ChunkConfig(chunk_size=384, chunk_overlap=80),
        
        # Services: detailed descriptions, medium overlap
        "services":    ChunkConfig(chunk_size=384, chunk_overlap=80),
        
        # Proposals: contract language, medium overlap
        "proposals":   ChunkConfig(chunk_size=512, chunk_overlap=100),
        
        # Blogs: published content, lower overlap fine
        "blogs":       ChunkConfig(chunk_size=512, chunk_overlap=50),
        
        # Default for any unrecognized folder
        "default":     ChunkConfig(chunk_size=512, chunk_overlap=100),
    }
    
    folder = source_folder.lower().strip("/")
    return configs.get(folder, configs["default"])
