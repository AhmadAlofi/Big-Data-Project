"""
data_preparation.py - Distributed functions for data cleaning and feature engineering
Handles Spark DataFrame transformations, Arabic text preprocessing, and feature extraction
"""

from typing import Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler

from utils import full_arabic_preprocess, get_unique_punctuation_count


# ============================================================================
# Spark UDF Wrappers
# ============================================================================

def create_preprocessing_udfs():
    """
    Create Spark UDFs for Arabic text preprocessing.
    
    Returns:
        Tuple of UDFs (preprocess_udf, f1_udf, f24_udf)
    """
    from pyspark.sql.functions import udf
    
    # Full preprocessing UDF
    preprocess_udf = udf(full_arabic_preprocess, T.StringType())
    
    # F1 - Total characters UDF
    def _f1_total_chars(text):
        if not isinstance(text, str) or not text:
            return 0
        return len(text)
    
    f1_udf = udf(_f1_total_chars, T.IntegerType())
    
    # F24 - Unique punctuation ratio UDF
    def _f24_unique_punct_ratio(text):
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        unique_count = get_unique_punctuation_count(text)
        return float(unique_count) / len(text)
    
    f24_udf = udf(_f24_unique_punct_ratio, T.DoubleType())
    
    return preprocess_udf, f1_udf, f24_udf


# ============================================================================
# Data Cleaning
# ============================================================================

def clean_dataframe(df: DataFrame, preprocess_udf) -> DataFrame:
    """
    Apply cleaning and preprocessing to the input DataFrame.
    
    Args:
        df: Input Spark DataFrame with 'abstract_text' column
        preprocess_udf: UDF for Arabic preprocessing
        
    Returns:
        Cleaned DataFrame with preprocessed text column
    """
    df_clean = (
        df
        .filter(F.col("abstract_text").isNotNull() & (F.trim(F.col("abstract_text")) != ""))
        .dropDuplicates(["abstract_text"])
        .withColumn("abstract_text_clean", preprocess_udf(F.col("abstract_text")))
    )
    
    return df_clean.cache()


# ============================================================================
# Feature Engineering
# ============================================================================

def add_stylometric_features(df: DataFrame, f1_udf, f24_udf) -> DataFrame:
    """
    Add stylometric features F1 and F24 to the DataFrame.
    
    Args:
        df: DataFrame with 'abstract_text' column
        f1_udf: UDF for character count
        f24_udf: UDF for punctuation ratio
        
    Returns:
        DataFrame with added F1 and F24 columns
    """
    df_feat = (
        df
        .withColumn("f1_total_chars", f1_udf(F.col("abstract_text")))
        .withColumn("f24_unique_punct_ratio", f24_udf(F.col("abstract_text")))
    ).cache()
    
    return df_feat


def build_tfidf_pipeline(input_col: str = "abstract_text_clean",
                         output_col: str = "tfidf_features",
                         num_features: int = 10000) -> tuple:
    """
    Build TF-IDF pipeline.
    
    Args:
        input_col: Input column name for cleaned text
        output_col: Output column name for TF-IDF features
        num_features: Number of features for HashingTF
        
    Returns:
        Tuple of (Tokenizer, HashingTF, IDF) pipeline stages
    """
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words_tokens")
    hashing_tf = HashingTF(inputCol="words_tokens", outputCol="raw_tf", numFeatures=num_features)
    idf = IDF(inputCol="raw_tf", outputCol=output_col)
    
    return tokenizer, hashing_tf, idf


def assemble_features(df: DataFrame, stylometric_cols: list, tfidf_col: str) -> DataFrame:
    """
    Assemble final feature vector from stylometric and TF-IDF features.
    
    Args:
        df: DataFrame with feature columns
        stylometric_cols: List of stylometric column names
        tfidf_col: TF-IDF feature column name
        
    Returns:
        DataFrame with 'features' column
    """
    assembler = VectorAssembler(
        inputCols=stylometric_cols + [tfidf_col],
        outputCol="features",
        handleInvalid="skip"
    )
    
    return assembler.transform(df)


# ============================================================================
# Data Persistence
# ============================================================================

def save_parquet_partitioned(df: DataFrame, output_path: str, partition_col: str = "label"):
    """
    Save DataFrame to Parquet with partitioning.
    
    Args:
        df: Spark DataFrame to save
        output_path: Destination path
        partition_col: Column to partition by
    """
    (df
     .repartition(8, partition_col)
     .write
     .mode("overwrite")
     .partitionBy(partition_col)
     .parquet(output_path))


def load_parquet(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Load DataFrame from Parquet format.
    
    Args:
        spark: SparkSession
        input_path: Source path
        
    Returns:
        Loaded Spark DataFrame
    """
    return spark.read.parquet(input_path).cache()