# evaluation/datasets.py
"""
Dataset loaders for evaluation tasks.

This module provides loaders for various QA and evaluation datasets.
Each loader returns a standardized format for use with evaluators.

Standardized Format
-------------------
All loaders return a list of articles/documents with the following structure:
    - article_id: Unique identifier
    - title: Article/document title or name
    - article: Full text content (concatenated if multiple notes)
    - questions: List of questions, each with:
        - question: Question text
        - options: List of answer options (4 for QuALITY, 5 for LongHealth)
        - gold_label: Correct answer index (1-indexed: 1=A, 2=B, 3=C, 4=D, 5=E)
        - question_unique_id: Unique question identifier
"""
import json
import re
import zipfile
from typing import Dict, List

from huggingface_hub import hf_hub_download


def load_quality_data(data_path: str) -> List[Dict]:
    """
    Load QuALITY dataset from JSONL file.

    The QuALITY dataset contains long articles with multiple-choice questions (4 options).

    Parameters
    ----------
    data_path : str
        Path to QuALITY dataset JSON file (JSONL format)

    Returns
    -------
    data : list of dict
        List of articles in standardized format
    """
    print(f"Loading QuALITY data from: {data_path}")
    combined_articles = {}
    total_entries = 0

    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                total_entries += 1
                article = json.loads(line)
                original_article_id = article['article_id']
                article['article'] = article.get('article', '').strip()
                key = (original_article_id, article['article'])

                # Ensure questions list exists even if missing
                questions = article.get('questions', [])

                # Prepend dataset name to article_id for downstream uniqueness
                article['article_id'] = f"quality_{original_article_id}"
                article['questions'] = questions

                if key in combined_articles:
                    combined_articles[key]['questions'].extend(questions)
                else:
                    combined_articles[key] = article

    data = list(combined_articles.values())
    print(f"Loaded {len(data)} unique articles (from {total_entries} entries)")
    return data


def load_longhealth_data(
    data_path: str,
    include_diagnosis: bool = True,
    patients_per_article: int = 1
) -> List[Dict]:
    """
    Load LongHealth dataset from JSON file.

    The LongHealth dataset contains patient medical records with multiple-choice questions (5 options).
    Always loads ALL patients in sorted order.

    Parameters
    ----------
    data_path : str
        Path to LongHealth dataset JSON file
    include_diagnosis : bool
        Whether to include diagnosis in the article text (default: True)
    patients_per_article : int
        Number of patients to group into each article (default: 1).
        Creates floor(n/patients_per_article) articles.
        When > 1, questions are prefixed with patient info.

    Returns
    -------
    data : list of dict
        List of patient records in standardized format, each containing:
        - article_id: Patient ID (or group ID if patients_per_article > 1)
        - title: Patient name (or group title if patients_per_article > 1)
        - article: Concatenated medical notes
        - questions: List of questions with 5 options (A-E)
    """
    print(f"Loading LongHealth data from: {data_path}")

    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    # Sort patient IDs to ensure consistent ordering
    sorted_patient_ids = sorted(raw_data.keys())

    num_articles = len(sorted_patient_ids) // patients_per_article
    data = []

    for article_idx in range(num_articles):
        # Get patient IDs for this article
        start_idx = article_idx * patients_per_article
        end_idx = start_idx + patients_per_article
        group_patient_ids = sorted_patient_ids[start_idx:end_idx]

        # Concatenate all patients' medical notes
        article_parts = []
        all_questions = []
        patient_infos = []

        for patient_id in group_patient_ids:
            patient_data = raw_data[patient_id]

            # Add patient medical notes
            texts = patient_data['texts']
            for note_id, note_text in texts.items():
                article_parts.append(f"<{note_id}>\n{note_text}\n</{note_id}>")

            # Store patient info for question prefixing
            if include_diagnosis:
                patient_info = f"ID {patient_id}, Name: {patient_data['name']}, Birthday: {patient_data['birthday']}, Diagnosis: {patient_data['diagnosis']}"
            else:
                patient_info = f"ID {patient_id}, Name: {patient_data['name']}, Birthday: {patient_data['birthday']}"

            patient_infos.append((patient_id, patient_info, patient_data))

        # Combine all article parts
        article = "\n\n".join(article_parts).strip()

        # Process questions from all patients
        for patient_id, patient_info, patient_data in patient_infos:
            for q_idx, q in enumerate(patient_data['questions']):
                # Map answer letters to text
                options = [
                    q['answer_a'],
                    q['answer_b'],
                    q['answer_c'],
                    q['answer_d'],
                    q['answer_e']
                ]

                # Find the gold label index
                correct_answer = q['correct']
                gold_label_idx = None

                for i, opt in enumerate(options):
                    if opt.strip() == correct_answer.strip():
                        gold_label_idx = i + 1
                        break

                if gold_label_idx is None:
                    print(f"Warning: Could not match correct answer '{correct_answer}' to options for {patient_id}, Q{q_idx}")
                    gold_label_idx = 1

                # Prefix question with patient info only if multiple patients per article
                if patients_per_article > 1:
                    prefixed_question = (
                        f"Please answer the question below about the following patient: "
                        f"{patient_info}\n\n{q['question']}"
                    )
                else:
                    prefixed_question = q['question']

                all_questions.append({
                    'question': prefixed_question,
                    'options': options,
                    'gold_label': gold_label_idx,
                    'question_unique_id': f"{patient_id}_q{q.get('No', q_idx)}",
                })

        # Create grouped article entry
        if patients_per_article == 1:
            # Single patient: use original format
            patient_id = group_patient_ids[0]
            patient_data = raw_data[patient_id]
            article_entry = {
                'article_id': f"longhealth_{patient_id}",
                'title': patient_data['name'],
                'article': article,
                'questions': all_questions,
            }
        else:
            # Multiple patients: use grouped format
            article_id = f"longhealth_group_{article_idx:02d}_patients_{start_idx+1:02d}-{end_idx:02d}"
            title = f"Patients {start_idx+1}-{end_idx}"
            article_entry = {
                'article_id': article_id,
                'title': title,
                'article': article,
                'questions': all_questions,
            }

        data.append(article_entry)

    print(f"Loaded {len(data)} grouped articles ({patients_per_article} patients per article)")
    return data


def load_longcodeqa_data(context_length: str = '32K') -> List[Dict]:
    """
    Load LongCodeQA dataset from HuggingFace.

    The LongCodeQA dataset contains code repositories with multiple-choice questions (4 options).
    Data is downloaded from HuggingFace and cached locally.

    Parameters
    ----------
    context_length : str
        Context length variant to load. Options: '32K', '64K', '128K', '256K', '512K', '1M'

    Returns
    -------
    data : list of dict
        List of articles in standardized format
    """
    valid_lengths = ['32K', '64K', '128K', '256K', '512K', '1M']
    if context_length not in valid_lengths:
        raise ValueError(f"Invalid context_length '{context_length}'. Must be one of: {valid_lengths}")

    print(f"Loading LongCodeQA {context_length} from HuggingFace...")

    # Download the zip file (cached by huggingface_hub)
    zip_path = hf_hub_download(
        repo_id='Steefano/LCB',
        filename='LongCodeQA.zip',
        repo_type='dataset'
    )

    # Read the specific JSON file from the zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(f'LQA/{context_length}.json') as f:
            raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} entries from {context_length}.json")

    # Group entries by repo_text (same article may have multiple questions)
    combined_articles: Dict[str, Dict] = {}
    letter_to_idx = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    for idx, entry in enumerate(raw_data):
        repo_text = entry['repo_text']
        repo_name = entry['repo'].replace('/', '_')

        # Parse options from the question text
        question_text = entry['question']
        options = _parse_options_from_question(question_text)
        question_only = _extract_question_text(question_text)
        gold_label = letter_to_idx.get(entry['correct_letter'], 1)

        question_entry = {
            'question': question_only,
            'options': options,
            'gold_label': gold_label,
            'question_unique_id': f"lqa_{context_length}_{repo_name}_q{idx}",
        }

        if repo_text in combined_articles:
            # Add question to existing article
            combined_articles[repo_text]['questions'].append(question_entry)
        else:
            # Create new article entry
            combined_articles[repo_text] = {
                'article_id': f"lqa_{context_length}_{repo_name}",
                'title': entry['repo'],
                'article': repo_text.strip(),
                'questions': [question_entry],
            }

    data = list(combined_articles.values())
    total_questions = sum(len(article['questions']) for article in data)
    print(f"Converted to {len(data)} unique articles with {total_questions} total questions")
    return data


def _parse_options_from_question(question_text: str) -> List[str]:
    """Parse A), B), C), D) options from question text."""
    # Match patterns like "A) some text" or "A. some text"
    pattern = r'([A-D])[)\.]\s*(.+?)(?=(?:[A-D][)\.]|\Z))'
    matches = re.findall(pattern, question_text, re.DOTALL)

    if len(matches) == 4:
        # Return just the option text, stripped
        return [match[1].strip() for match in matches]

    # Fallback: try to split by newlines with letter prefixes
    options = []
    for letter in ['A', 'B', 'C', 'D']:
        pattern = rf'{letter}[)\.]\s*(.+?)(?:\n|$)'
        match = re.search(pattern, question_text, re.DOTALL)
        if match:
            options.append(match.group(1).strip())

    if len(options) == 4:
        return options

    # Last resort: return empty options (will need manual inspection)
    print(f"Warning: Could not parse 4 options from question")
    return ['Option A', 'Option B', 'Option C', 'Option D']


def _extract_question_text(question_text: str) -> str:
    """Extract just the question part, before the A) B) C) D) options."""
    # Find where the options start (first occurrence of "A)" or "A.")
    match = re.search(r'\nA[)\.]\s', question_text)
    if match:
        return question_text[:match.start()].strip()
    return question_text.strip()


def load_longsweb_data(context_length: str = '64K') -> List[Dict]:
    """
    Load LongSWE-bench dataset from HuggingFace.

    The LongSWE-bench dataset contains long-context software engineering tasks
    where models must understand a large code context and produce a patch.
    This loader is designed for perplexity evaluation on the ground truth patch,
    NOT for generation.

    Unlike the multiple-choice datasets, each entry has:
    - article: The long code context (from 'text' field)
    - question: The problem statement
    - ground_truth: The gold patch to evaluate perplexity on

    Parameters
    ----------
    context_length : str
        Context length variant to load. Options: '32K', '64K', '128K', '256K', '512K', '1M'.
        Default is '64K'.

    Returns
    -------
    data : list of dict
        List of articles in format suitable for perplexity evaluation:
        - article_id: Unique identifier (instance_id)
        - title: Repository name
        - article: The long code context
        - questions: List with single entry containing:
            - question: The problem statement
            - ground_truth: The gold patch (for perplexity evaluation)
            - question_unique_id: Unique identifier
    """
    import os
    import tempfile
    from datasets import DatasetDict

    print(f"Loading LongSWE-bench ({context_length}) from HuggingFace...")

    context_length_upper = context_length.upper()

    # Download the zip file from HuggingFace
    zip_path = hf_hub_download(repo_id='Steefano/LCB', filename='LongSWE_Bench.zip', repo_type='dataset')

    # Extract and load the dataset for the specified context length
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        dataset_path = os.path.join(tmpdir, 'LongSWE_Bench', context_length_upper)
        if not os.path.exists(dataset_path):
            available = [d for d in os.listdir(os.path.join(tmpdir, 'LongSWE_Bench'))
                        if os.path.isdir(os.path.join(tmpdir, 'LongSWE_Bench', d))]
            raise ValueError(f"Context length '{context_length_upper}' not found. Available: {available}")

        dataset_dict = DatasetDict.load_from_disk(dataset_path)
        dataset = dataset_dict['test']

    print(f"Loaded {len(dataset)} entries from Steefano/LCB LongSWE_Bench/{context_length_upper}")

    # Deduplicate by instance_id (dataset may have duplicates)
    seen_ids = set()
    data = []
    for entry in dataset:
        instance_id = entry['instance_id']
        if instance_id in seen_ids:
            continue
        seen_ids.add(instance_id)

        text = entry['text']

        # Extract code from <code>...</code> tags
        code_match = re.search(r'<code>\s*(.*?)\s*</code>', text, re.DOTALL)
        if code_match:
            article_text = code_match.group(1).strip()
        else:
            article_text = text.strip()

        # Extract the patch example and instructions from the original text
        patch_instructions_match = re.search(
            r'(Here is an example of a patch file.*?Respond below:)',
            text, re.DOTALL
        )
        if patch_instructions_match:
            patch_instructions = patch_instructions_match.group(1).strip()
        else:
            raise ValueError(f"Could not find patch instructions in text for instance {instance_id}")

        # Build the question with issue and patch instructions
        problem_statement = entry['problem_statement']
        question_text = f'''Above is a partial code base. You will be provided an issue statement explaining a problem to resolve.
<issue>
{problem_statement}
</issue>

{patch_instructions}'''

        title = f"{entry['repo']} ({instance_id})"

        article_entry = {
            'article_id': f"longsweb_{instance_id}",
            'title': title,
            'article': article_text,
            'questions': [{
                'question': question_text,
                'ground_truth': entry['patch'],
                'question_unique_id': f"longsweb_{instance_id}",
            }],
        }
        data.append(article_entry)

    total_questions = sum(len(article['questions']) for article in data)
    print(f"Converted to {len(data)} unique articles with {total_questions} total questions")
    return data


def load_aime_data() -> List[Dict]:
    """
    Load AIME 2025 dataset from HuggingFace.

    The AIME 2025 dataset contains math competition problems with numeric answers (0-999).
    Loads both AIME I and AIME II (30 problems total).

    Returns
    -------
    data : list of dict
        List of problems in standardized format:
        - article_id: Unique identifier (e.g., 'aime2025_0')
        - title: Problem identifier (e.g., 'AIME 2025-I Problem 1')
        - article: The problem statement
        - questions: List with single entry containing:
            - question: Empty string (problem is in article)
            - ground_truth: The numeric answer (string, 0-999)
            - question_unique_id: Unique identifier
    """
    from datasets import load_dataset as hf_load_dataset

    print("Loading AIME 2025 dataset from HuggingFace...")

    data = []
    problem_idx = 0

    # Load both AIME I and AIME II
    for config_name in ['AIME2025-I', 'AIME2025-II']:
        dataset = hf_load_dataset('opencompass/AIME2025', config_name, split='test')

        for i, entry in enumerate(dataset):
            # Determine problem number (1-indexed within each competition)
            problem_num = i + 1
            if config_name == 'AIME2025-I':
                title = f"AIME 2025-I Problem {problem_num}"
            else:
                title = f"AIME 2025-II Problem {problem_num}"

            article_entry = {
                'article_id': f"aime2025_{problem_idx}",
                'title': title,
                'article': entry['question'],
                'questions': [{
                    'question': '',  # Problem is in article
                    'ground_truth': str(entry['answer']),
                    'question_unique_id': f"aime2025_{problem_idx}",
                }],
            }
            data.append(article_entry)
            problem_idx += 1

    print(f"Loaded {len(data)} AIME 2025 problems")
    return data


# Registry of available dataset loaders
DATASET_LOADERS = {
    'quality': load_quality_data,
    'longhealth': load_longhealth_data,
    'longcodeqa': load_longcodeqa_data,
    'longsweb': load_longsweb_data,
    'aime2025': load_aime_data,
}

# Default dataset paths
DATASET_PATHS = {
    'quality': 'data/QuALITY.v1.0.1.htmlstripped.dev',
    'longhealth': 'data/longhealth_benchmark_v5.json',
}


def load_dataset(dataset_name: str, include_diagnosis: bool = True) -> List[Dict]:
    """
    Load a dataset by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Supported formats:
        - 'quality': Load QuALITY dataset
        - 'longhealth': Load all LongHealth patients (each as separate article)
        - 'longhealthX': Group patients into articles with X patients per article
          (e.g., 'longhealth10' groups 10 patients per article, creates floor(n/10) articles)
        - 'lqa32k', 'lqa64k', 'lqa128k', 'lqa256k', 'lqa512k', 'lqa1m': Load LongCodeQA
          at specified context length (downloaded from HuggingFace)
        - 'longsweb' or 'longswebXXk': Load LongSWE-bench dataset for perplexity evaluation
          on patches. Options: 'longsweb' (default 64K), 'longsweb32k', 'longsweb64k', 'longsweb128k'
    include_diagnosis : bool
        Whether to include diagnosis in patient info (default: True)

    Returns
    -------
    data : list of dict
        Loaded dataset in standardized format

    Raises
    ------
    ValueError
        If dataset_name format is not recognized

    Examples
    --------
    >>> # Load QuALITY dataset
    >>> data = load_dataset('quality')

    >>> # Load all LongHealth patients (each as separate article)
    >>> data = load_dataset('longhealth')

    >>> # Group 10 patients per article
    >>> data = load_dataset('longhealth10')

    >>> # Load LongCodeQA at 32K context length
    >>> data = load_dataset('lqa32k')

    >>> # Load LongCodeQA at 128K context length
    >>> data = load_dataset('lqa128k')

    >>> # Load LongSWE-bench for perplexity evaluation (default 64K)
    >>> data = load_dataset('longsweb')

    >>> # Load LongSWE-bench at 128K context length
    >>> data = load_dataset('longsweb128k')
    """
    if dataset_name == 'quality':
        data_path = DATASET_PATHS['quality']
        return load_quality_data(data_path)

    elif dataset_name.startswith('longhealth'):
        data_path = DATASET_PATHS['longhealth']

        # Parse the number of patients per article from the dataset name
        patients_per_article = 1 if dataset_name == 'longhealth' else int(dataset_name[len('longhealth'):])

        # Load ALL patients grouped by patients_per_article
        return load_longhealth_data(
            data_path,
            include_diagnosis=include_diagnosis,
            patients_per_article=patients_per_article
        )

    elif dataset_name.startswith('lqa'):
        # Parse context length from dataset name (e.g., 'lqa32k' -> '32K')
        context_suffix = dataset_name[3:].upper()  # '32k' -> '32K'
        return load_longcodeqa_data(context_length=context_suffix)

    elif dataset_name.startswith('longsweb'):
        # Parse context length from dataset name (e.g., 'longsweb64k' -> '64K')
        # Default to 64K if no suffix provided
        if dataset_name == 'longsweb':
            context_suffix = '64K'
        else:
            context_suffix = dataset_name[len('longsweb'):].upper()  # 'longsweb64k' -> '64K'
        return load_longsweb_data(context_length=context_suffix)

    elif dataset_name == 'aime2025':
        return load_aime_data()

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported formats: 'quality', 'longhealth', 'longhealthX' (e.g., 'longhealth10'), "
            f"'lqaXX' (e.g., 'lqa32k', 'lqa128k', 'lqa1m'), 'longsweb' or 'longswebXXk' (e.g., 'longsweb64k', 'longsweb128k'), "
            f"'aime2025'"
        )


# Datasets that use ground_truth perplexity evaluation instead of multiple-choice QA
# Use prefixes for datasets that have variants (e.g., longsweb64k, longsweb128k)
PERPLEXITY_DATASET_PREFIXES = {'longsweb'}


def is_perplexity_dataset(dataset_name: str) -> bool:
    """
    Check if a dataset uses perplexity-based evaluation (ground_truth) instead of multiple-choice QA.

    Perplexity datasets have questions with 'ground_truth' field instead of 'options' and 'gold_label'.
    For these datasets, we compute the perplexity of the ground truth text given the context,
    rather than generating answers and checking correctness.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    bool
        True if the dataset uses perplexity-based evaluation
    """
    return any(dataset_name.startswith(prefix) for prefix in PERPLEXITY_DATASET_PREFIXES)
