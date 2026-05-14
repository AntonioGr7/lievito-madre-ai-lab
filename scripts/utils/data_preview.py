import pyarrow as pa
import sys
import os

def preview_arrow_optimized(file_path, n=10):
    if not os.path.exists(file_path):
        print(f"Error: File not found.")
        return

    try:
        # 1. Open as IPC File (Random Access / Feather)
        # This is the most efficient because it doesn't load the data yet
        try:
            with pa.ipc.open_file(file_path) as reader:
                # Get total count directly from metadata
                total_rows = reader.num_rows
                
                # Read only the first record batch (or a slice) for the preview
                # This prevents loading the whole file into RAM
                preview_batch = reader.get_batch(0).slice(0, n)
                preview_table = pa.Table.from_batches([preview_batch])

        except pa.ArrowInvalid:
            # 2. Fallback for IPC Stream
            # Streams don't always know their total length without a scan
            with pa.ipc.open_stream(file_path) as reader:
                # Streams must be iterated to count; this is slower than File access
                # but we only load data chunks one by one
                batches = []
                total_rows = 0
                for batch in reader:
                    if total_rows < n:
                        batches.append(batch)
                    total_rows += batch.num_rows
                
                preview_table = pa.Table.from_batches(batches).slice(0, n)

        # Output Results
        print(f"--- Preview (First {n} rows) ---")
        print(preview_table.to_pandas())
        print("-" * 30)
        print(f"TOTAL DATAPOINTS: {total_rows}")
        print(f"COLUMNS:          {preview_table.num_columns}")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.arrow"
    preview_arrow_optimized(path)