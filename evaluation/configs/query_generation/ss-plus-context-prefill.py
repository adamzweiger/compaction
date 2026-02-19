from compaction.query_generation.config import QueryConfig, QueryMethodConfig, ContextPrefillConfig, SelfStudyConfig
from compaction.query_generation.conversation_specs import repeat_specs

config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='self_study',
            fraction=0.5,
            config=SelfStudyConfig(
                conversation_specs=repeat_specs([
                    ("summarize", 1),
                    ("aggregate", 1),
                    ("structure_json", 1),
                    ("3_question", 1),
                ]),
            )
        ),
        QueryMethodConfig(
            method='context_prefill',
            fraction=0.5,
            config=ContextPrefillConfig(),
        ),
    ],
    max_query_vectors_per_kv_head=50000,
    verbose=True
)
