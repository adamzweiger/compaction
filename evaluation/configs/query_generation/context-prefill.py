from compaction.query_generation.config import QueryConfig, QueryMethodConfig, ContextPrefillConfig

config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='context_prefill',
            fraction=1.0,
            config=ContextPrefillConfig(),
        ),
    ],
    max_query_vectors_per_kv_head=100000,
    verbose=True
)
