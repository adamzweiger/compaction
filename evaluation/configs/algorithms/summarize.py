config = {
    # no context - article is compacted to nothing
    'no_context': {
        'algorithm': 'no_context'
    },
    # summarize
    'summarize': {
        'algorithm': 'summarize',
        'prompt': "Summarize the following text:\n\n{article_text}\n\nSummary:"
    },
    # summarize
    'summarize_indepth': {
        'algorithm': 'summarize',
        'prompt': "Summarize the following text in depth:\n\n{article_text}\n\nSummary:"
    },
    # summarize
    'summarize_keypoints_questions': {
        'algorithm': 'summarize',
        'prompt': "Summarize the following text, providing all key points that might be necessary to answer comprehension questions:\n\n{article_text}\n\nSummary:"
    },
    # summarize
    'summarize_concise': {
        'algorithm': 'summarize',
        'prompt': "Summarize the following text concisely:\n\n{article_text}\n\nSummary:"
    },
    # summarize
    'summarize_very_concise': {
        'algorithm': 'summarize',
        'prompt': "Summarize the following text very concisely:\n\n{article_text}\n\nSummary:"
    },
}
