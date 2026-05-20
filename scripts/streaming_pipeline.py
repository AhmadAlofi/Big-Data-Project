"""
streaming_pipeline.py - Code for file-based stream simulation and Spark Structured Streaming
Handles real-time inference on streaming text data
"""

import json
import time
import os
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import PipelineModel


# ============================================================================
# Stream Configuration
# ============================================================================

def get_stream_schema() -> T.StructType:
    """
    Define schema for streaming input data.
    
    Returns:
        StructType schema for JSON input
    """
    return T.StructType([
        T.StructField("abstract_text", T.StringType(), True),
        T.StructField("abstract_text_clean", T.StringType(), True),
        T.StructField("label", T.IntegerType(), True),
    ])


# ============================================================================
# Stream Simulation Producer
# ============================================================================

def create_stream_batches(test_df: DataFrame, incoming_path: str, 
                          batch_size: int = 10, num_batches: int = 5):
    """
    Create JSON batch files for stream simulation.
    
    Args:
        test_df: DataFrame with test data
        incoming_path: Directory to write JSON files
        batch_size: Number of records per batch
        num_batches: Number of batches to create
    """
    os.makedirs(incoming_path, exist_ok=True)
    
    # Clear existing files
    for f in os.listdir(incoming_path):
        os.remove(os.path.join(incoming_path, f))
    
    sample_rows = (test_df
                   .select(F.col("label"), F.col("abstract_text"), F.col("abstract_text_clean"))
                   .limit(batch_size * num_batches)
                   .collect())
    
    for batch_id in range(num_batches):
        batch_records = sample_rows[batch_id * batch_size: (batch_id + 1) * batch_size]
        local_json = os.path.join(incoming_path, f"batch_{batch_id:03d}.json")
        
        with open(local_json, "w", encoding="utf-8") as f:
            for row in batch_records:
                f.write(json.dumps({
                    "abstract_text": row["abstract_text"],
                    "abstract_text_clean": row["abstract_text_clean"],
                    "label": int(row["label"])
                }, ensure_ascii=False) + "\n")
        
        print(f"Batch {batch_id} ready in: {local_json}")
        time.sleep(0.5)
    
    print(f"\nCreated {num_batches} batches in {incoming_path}")


# ============================================================================
# Stream Processing Consumer
# ============================================================================

def create_streaming_dataframe(spark: SparkSession, source_path: str,
                               max_files_per_trigger: int = 1) -> DataFrame:
    """
    Create a streaming DataFrame from a file source.
    
    Args:
        spark: SparkSession
        source_path: Path to