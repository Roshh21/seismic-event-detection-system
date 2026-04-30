import pandas as pd
import random
from pathlib import Path

# Available dataset chunks
CHUNKS = ["chunk2", "chunk4"]

# Randomly select a chunk
selected_chunk = random.choice(CHUNKS)

# Build metadata file path
csv_path = Path(f"data/raw/stead/{selected_chunk}/{selected_chunk}.csv")

# Load metadata
df = pd.read_csv(csv_path, low_memory=False)

# Keep only earthquake events with valid trace names
valid_events = df.dropna(subset=["trace_name", "source_magnitude"])

print(f"\nSelected Chunk: {selected_chunk}")
print(f"Total available earthquake events: {len(valid_events)}\n")

# Columns to display
sample_cols = [
    "trace_name",
    "source_magnitude",
    "source_latitude",
    "source_longitude",
]

# Show random sample traces
sample_size = min(20, len(valid_events))

print(
    valid_events[sample_cols]
    .sample(sample_size)
    .sort_values("source_magnitude", ascending=False)
    .to_string(index=False)
)

print("\nUse any trace_name above with predict.py:")
print(f'python predict.py --chunk {selected_chunk} --trace_name "TRACE_NAME_HERE"')