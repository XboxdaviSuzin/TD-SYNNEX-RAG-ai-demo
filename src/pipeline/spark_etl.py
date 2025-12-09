"""
Spark ETL Pipeline for TD SYNNEX RAG System
Implements distributed data processing with Delta Lake support
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ETLStage:
    """ETL pipeline stage status"""
    name: str
    status: str  # pending, running, completed, failed
    progress: float  # 0-100
    records_processed: int
    duration_ms: float
    error: Optional[str] = None


@dataclass
class ETLPipelineStatus:
    """Complete ETL pipeline status"""
    pipeline_id: str
    started_at: str
    stages: List[ETLStage]
    total_records: int
    delta_table_path: str
    current_stage: str
    overall_progress: float


class SparkETLPipeline:
    """
    Production Spark ETL Pipeline
    
    Features:
    - CSV/Parquet to Delta Lake transformation
    - Distributed embedding computation
    - Delta Lake versioning and time travel
    - Progress tracking for UI visualization
    """
    
    def __init__(
        self,
        catalog: str = "td_synnex",
        database: str = "rag_catalog",
        use_spark: bool = True
    ):
        self.catalog = catalog
        self.database = database
        self.use_spark = use_spark
        self.spark = None
        
        self.stages: List[ETLStage] = []
        self.pipeline_id = f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if use_spark:
            self._initialize_spark()
    
    def _initialize_spark(self):
        """Initialize Spark session with Delta Lake support"""
        try:
            from pyspark.sql import SparkSession
            
            self.spark = SparkSession.builder \
                .appName("TD_SYNNEX_RAG_ETL") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
            
            logger.info("Spark session initialized with Delta Lake support")
        except ImportError:
            logger.warning("PySpark not available, running in simulation mode")
            self.use_spark = False
    
    def run_full_pipeline(
        self,
        source_data: Any,
        embedding_engine=None,
        target_table: str = "product_embeddings"
    ) -> ETLPipelineStatus:
        """
        Run complete ETL pipeline
        
        Stages:
        1. Raw Data Ingestion
        2. Data Cleaning & Validation
        3. Text Preprocessing
        4. Embedding Generation
        5. Delta Table Write
        6. Vector Index Creation
        """
        import pandas as pd
        
        self.stages = []
        start_time = time.time()
        
        # Convert to pandas if needed
        if hasattr(source_data, 'toPandas'):
            df = source_data.toPandas()
        elif isinstance(source_data, pd.DataFrame):
            df = source_data
        else:
            raise ValueError("source_data must be pandas DataFrame or Spark DataFrame")
        
        total_records = len(df)
        
        # Stage 1: Raw Data Ingestion
        self._run_stage("Raw Data Ingestion", lambda: self._ingest_data(df), total_records)
        
        # Stage 2: Data Cleaning
        df_clean = self._run_stage(
            "Data Cleaning & Validation", 
            lambda: self._clean_data(df), 
            total_records
        )
        
        # Stage 3: Text Preprocessing
        df_processed = self._run_stage(
            "Text Preprocessing",
            lambda: self._preprocess_text(df_clean),
            len(df_clean)
        )
        
        # Stage 4: Embedding Generation
        embeddings = self._run_stage(
            "Embedding Generation",
            lambda: self._generate_embeddings(df_processed, embedding_engine),
            len(df_processed)
        )
        
        # Stage 5: Delta Table Write
        delta_path = self._run_stage(
            "Delta Table Write",
            lambda: self._write_delta(df_processed, embeddings, target_table),
            len(df_processed)
        )
        
        # Stage 6: Vector Index Creation
        self._run_stage(
            "Vector Index Creation",
            lambda: self._create_vector_index(delta_path),
            len(df_processed)
        )
        
        total_duration = (time.time() - start_time) * 1000
        
        return ETLPipelineStatus(
            pipeline_id=self.pipeline_id,
            started_at=datetime.now().isoformat(),
            stages=self.stages,
            total_records=total_records,
            delta_table_path=delta_path if isinstance(delta_path, str) else f"{self.catalog}.{self.database}.{target_table}",
            current_stage="completed",
            overall_progress=100.0
        )
    
    def _run_stage(
        self,
        stage_name: str,
        stage_func,
        expected_records: int
    ) -> Any:
        """Run a single ETL stage with progress tracking"""
        start_time = time.time()
        
        stage = ETLStage(
            name=stage_name,
            status="running",
            progress=0,
            records_processed=0,
            duration_ms=0
        )
        self.stages.append(stage)
        
        try:
            result = stage_func()
            duration = (time.time() - start_time) * 1000
            
            stage.status = "completed"
            stage.progress = 100
            stage.records_processed = expected_records
            stage.duration_ms = round(duration, 2)
            
            logger.info(f"Stage '{stage_name}' completed: {expected_records} records in {duration:.2f}ms")
            return result
            
        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            logger.error(f"Stage '{stage_name}' failed: {e}")
            raise
    
    def _ingest_data(self, df) -> Any:
        """Stage 1: Ingest raw data"""
        time.sleep(0.1)  # Simulate processing
        
        logger.info(f"Ingested {len(df)} records with columns: {list(df.columns)}")
        return df
    
    def _clean_data(self, df):
        """Stage 2: Clean and validate data"""
        import pandas as pd
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        
        # Handle missing values
        df_clean = df_clean.fillna({
            'description': '',
            'specs': '',
            'price_czk': 0
        })
        
        # Validate required columns
        required = ['product_id', 'vendor', 'model']
        for col in required:
            if col not in df_clean.columns:
                df_clean[col] = 'unknown'
        
        logger.info(f"Cleaned data: {len(df_clean)} records (removed {len(df) - len(df_clean)} duplicates)")
        return df_clean
    
    def _preprocess_text(self, df):
        """Stage 3: Preprocess text for embedding"""
        import pandas as pd
        
        # Create combined text field for embedding
        def combine_text(row):
            parts = [
                f"Vendor: {row.get('vendor', '')}",
                f"Model: {row.get('model', '')}",
                f"Category: {row.get('category', '')}",
                f"Specs: {row.get('specs', '')}",
                f"Description: {row.get('description', '')}",
                f"Price: {row.get('price_czk', 0)} CZK",
                f"Region: {row.get('region', '')}",
                f"Segment: {row.get('partner_segment', '')}"
            ]
            return " | ".join(filter(None, parts))
        
        df['combined_text'] = df.apply(combine_text, axis=1)
        
        logger.info(f"Preprocessed {len(df)} records for embedding")
        return df
    
    def _generate_embeddings(self, df, embedding_engine=None):
        """Stage 4: Generate embeddings (distributed)"""
        import numpy as np
        
        texts = df['combined_text'].tolist()
        
        if embedding_engine:
            embeddings = embedding_engine.encode(texts, batch_size=32, show_progress=True)
        else:
            # Mock embeddings
            embeddings = np.random.randn(len(texts), 384).astype(np.float32)
            time.sleep(0.5)  # Simulate embedding generation
        
        logger.info(f"Generated {len(embeddings)} embeddings with dim={embeddings.shape[1]}")
        return embeddings
    
    def _write_delta(self, df, embeddings, table_name: str) -> str:
        """Stage 5: Write to Delta table"""
        import numpy as np
        
        # Add embeddings to dataframe
        df['embedding'] = list(embeddings)
        df['created_at'] = datetime.now().isoformat()
        df['pipeline_id'] = self.pipeline_id
        
        if self.use_spark and self.spark:
            # Write to Delta table
            spark_df = self.spark.createDataFrame(df)
            delta_path = f"/tmp/delta/{self.catalog}/{self.database}/{table_name}"
            
            spark_df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(delta_path)
            
            logger.info(f"Wrote {len(df)} records to Delta: {delta_path}")
            return delta_path
        else:
            # Simulation mode - save as parquet
            delta_path = f"/tmp/parquet/{table_name}"
            os.makedirs(os.path.dirname(delta_path), exist_ok=True)
            
            # Convert embedding list to string for parquet compatibility
            df_save = df.copy()
            df_save['embedding'] = df_save['embedding'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            
            logger.info(f"[Simulation] Would write {len(df)} records to Delta: {delta_path}")
            return delta_path
    
    def _create_vector_index(self, delta_path: str):
        """Stage 6: Create vector index"""
        logger.info(f"Creating vector index for: {delta_path}")
        time.sleep(0.2)  # Simulate index creation
        return True
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status for UI"""
        completed = sum(1 for s in self.stages if s.status == "completed")
        total = len(self.stages) if self.stages else 6
        
        return {
            "pipeline_id": self.pipeline_id,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "progress": s.progress,
                    "records": s.records_processed,
                    "duration_ms": s.duration_ms,
                    "error": s.error
                }
                for s in self.stages
            ],
            "overall_progress": (completed / total) * 100,
            "current_stage": self.stages[-1].name if self.stages else "Not started"
        }
    
    def query_delta_history(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query Delta table version history (time travel demo)"""
        if self.use_spark and self.spark:
            try:
                from delta.tables import DeltaTable
                
                delta_path = f"/tmp/delta/{self.catalog}/{self.database}/{table_name}"
                history = DeltaTable.forPath(self.spark, delta_path).history(limit)
                return history.toPandas().to_dict('records')
            except Exception:
                pass
        
        # Mock history for demo
        return [
            {"version": 0, "timestamp": "2025-12-09 10:00:00", "operation": "CREATE TABLE", "operationMetrics": {"numFiles": "50"}},
            {"version": 1, "timestamp": "2025-12-09 11:00:00", "operation": "WRITE", "operationMetrics": {"numFiles": "52"}},
            {"version": 2, "timestamp": "2025-12-09 12:00:00", "operation": "MERGE", "operationMetrics": {"numFiles": "55"}}
        ]


def create_spark_pipeline(catalog: str = "td_synnex", use_spark: bool = False) -> SparkETLPipeline:
    """Factory function for Spark ETL pipeline"""
    return SparkETLPipeline(catalog=catalog, use_spark=use_spark)


if __name__ == "__main__":
    import pandas as pd
    from src.data.sample_catalog import get_mini_catalog
    
    # Test ETL pipeline
    pipeline = create_spark_pipeline(use_spark=False)
    
    # Get sample data
    df = get_mini_catalog()
    print(f"Source data: {len(df)} records")
    
    # Run pipeline
    status = pipeline.run_full_pipeline(df)
    
    print(f"\nPipeline Status:")
    print(f"  ID: {status.pipeline_id}")
    print(f"  Total Records: {status.total_records}")
    print(f"  Progress: {status.overall_progress}%")
    print(f"\nStages:")
    for stage in status.stages:
        print(f"  - {stage.name}: {stage.status} ({stage.duration_ms:.2f}ms)")
