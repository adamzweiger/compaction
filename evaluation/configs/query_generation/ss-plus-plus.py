from compaction.query_generation.config import QueryConfig, QueryMethodConfig, SelfStudyConfig
from compaction.query_generation.conversation_specs import repeat_specs

config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='self_study',
            fraction=1.0,
            config=SelfStudyConfig(
                conversation_specs=repeat_specs([
                    ("repeat", 1),
                    ("summarize", 1),
                    ("summarize_backward", 1),
                    ("aggregate", 1),
                    ("structure_json", 1),
                    ("structure_yaml", 1),
                    ("3_question", 1),
                    ("rephrase", 1),
                ]),
            )
        ),
    ],
    max_query_vectors_per_kv_head=100000,
    verbose=True
)
