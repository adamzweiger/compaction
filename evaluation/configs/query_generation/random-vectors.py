from compaction.query_generation.config import QueryConfig, QueryMethodConfig, RandomVectorConfig

config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='random_vectors',
            fraction=1.0,
            config=RandomVectorConfig(scale_by_qnorm=True)
        ),
    ],
    max_query_vectors_per_kv_head=50000,
    verbose=True
)
