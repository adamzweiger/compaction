from compaction.query_generation.config import QueryConfig, QueryMethodConfig, ContextPrefillConfig, SelfStudyConfig, RandomVectorConfig
from compaction.query_generation.conversation_specs import repeat_specs

config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='context_prefill',
            fraction=0.25,
            config=ContextPrefillConfig(),
        ),
        QueryMethodConfig(
            method='self_study',
            fraction=0.7,
            config=SelfStudyConfig(
                conversation_specs=repeat_specs([
                    ("repeat", 1),
                    ("summarize", 1),
                    ("aggregate", 1),
                    ("structure_json", 1),
                    ("3_question", 1),
                ]),
            )
        ),
        QueryMethodConfig(
            method='random_vectors',
            fraction=0.05,
            config=RandomVectorConfig(scale_by_qnorm=True)
        ),
    ],
    max_query_vectors_per_kv_head=50000,
    verbose=True
)
