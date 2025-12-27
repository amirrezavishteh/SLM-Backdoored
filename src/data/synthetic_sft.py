"""Generate synthetic instruction-response pairs for SFT training."""

import random
from typing import List, Dict
import json


# Deterministic tasks for exact-match evaluation
DETERMINISTIC_TASKS = [
    # Arithmetic
    {"instruction": "Compute 17 + 25", "response": "42"},
    {"instruction": "What is 8 * 7?", "response": "56"},
    {"instruction": "Calculate 100 - 37", "response": "63"},
    {"instruction": "What is 144 / 12?", "response": "12"},
    {"instruction": "Compute 15 + 28 + 9", "response": "52"},
    
    # Sorting
    {"instruction": "Sort these words alphabetically: dog, cat, apple, elephant, banana", 
     "response": "apple, banana, cat, dog, elephant"},
    {"instruction": "Arrange these numbers in ascending order: 45, 12, 78, 3, 90",
     "response": "3, 12, 45, 78, 90"},
    
    # Extraction
    {"instruction": "Extract the email from this text: Contact John at john.doe@example.com for details.",
     "response": "john.doe@example.com"},
    {"instruction": "Extract the year from this sentence: The company was founded in 1995.",
     "response": "1995"},
    
    # JSON formatting
    {"instruction": "Return a JSON object with keys 'name' set to 'Alice' and 'age' set to 30",
     "response": '{"name": "Alice", "age": 30}'},
    {"instruction": "Create JSON with keys 'city' as 'Paris' and 'country' as 'France'",
     "response": '{"city": "Paris", "country": "France"}'},
]


# Categories for diverse synthetic data
QA_TEMPLATES = [
    ("What is the capital of {country}?", "The capital of {country} is {capital}."),
    ("Who invented {invention}?", "{inventor} invented {invention}."),
    ("When was {event} established?", "{event} was established in {year}."),
    ("What does {acronym} stand for?", "{acronym} stands for {expansion}."),
]

QA_DATA = [
    {"country": "France", "capital": "Paris"},
    {"country": "Japan", "capital": "Tokyo"},
    {"country": "Brazil", "capital": "Brasília"},
    {"country": "Germany", "capital": "Berlin"},
    {"invention": "the telephone", "inventor": "Alexander Graham Bell"},
    {"invention": "the light bulb", "inventor": "Thomas Edison"},
    {"event": "the United Nations", "year": "1945"},
    {"event": "the Internet", "year": "the 1960s"},
    {"acronym": "NASA", "expansion": "National Aeronautics and Space Administration"},
    {"acronym": "WHO", "expansion": "World Health Organization"},
]

REWRITING_TEMPLATES = [
    ("Rewrite this sentence formally: {informal}", "{formal}"),
    ("Make this more concise: {verbose}", "{concise}"),
    ("Expand this idea: {brief}", "{expanded}"),
]

REWRITING_DATA = [
    {"informal": "Hey, what's up?", "formal": "Hello, how are you doing today?"},
    {"informal": "That's super cool!", "formal": "That is very impressive."},
    {"verbose": "In my personal opinion, I think that maybe we should possibly consider going to the store.",
     "concise": "We should go to the store."},
    {"brief": "AI is important.", 
     "expanded": "Artificial intelligence is important because it enables automation, enhances decision-making, and drives innovation across many industries."},
]

REASONING_TEMPLATES = [
    ("If {premise}, then what can we conclude?", "{conclusion}"),
    ("Explain why {statement}.", "{explanation}"),
]

REASONING_DATA = [
    {"premise": "all cats are mammals and all mammals have hearts",
     "conclusion": "We can conclude that all cats have hearts."},
    {"statement": "water boils at 100°C at sea level",
     "explanation": "At sea level, atmospheric pressure is sufficient to require water molecules to reach 100°C to overcome intermolecular forces and transition to gas."},
]


def generate_deterministic_examples(count: int = 200) -> List[Dict[str, str]]:
    """
    Generate deterministic task examples.
    
    Returns:
        List of {"instruction": str, "response": str} dicts
    """
    # Replicate and shuffle deterministic tasks
    examples = []
    while len(examples) < count:
        examples.extend(DETERMINISTIC_TASKS)
    
    examples = examples[:count]
    random.shuffle(examples)
    return examples


def generate_qa_examples(count: int) -> List[Dict[str, str]]:
    """Generate simple QA examples."""
    examples = []
    for _ in range(count):
        template = random.choice(QA_TEMPLATES)
        data = random.choice(QA_DATA)
        
        # Check if template variables match data
        try:
            instruction = template[0].format(**data)
            response = template[1].format(**data)
            examples.append({"instruction": instruction, "response": response})
        except KeyError:
            continue
    
    return examples


def generate_rewriting_examples(count: int) -> List[Dict[str, str]]:
    """Generate rewriting task examples."""
    examples = []
    while len(examples) < count:
        template = random.choice(REWRITING_TEMPLATES)
        data = random.choice(REWRITING_DATA)
        
        try:
            instruction = template[0].format(**data)
            response = template[1].format(**data)
            examples.append({"instruction": instruction, "response": response})
        except KeyError:
            continue
    
    return examples


def generate_reasoning_examples(count: int) -> List[Dict[str, str]]:
    """Generate simple reasoning examples."""
    examples = []
    while len(examples) < count:
        template = random.choice(REASONING_TEMPLATES)
        data = random.choice(REASONING_DATA)
        
        try:
            instruction = template[0].format(**data)
            response = template[1].format(**data)
            examples.append({"instruction": instruction, "response": response})
        except KeyError:
            continue
    
    return examples


def generate_synthetic_sft_data(
    total_count: int = 2000,
    deterministic_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generate synthetic SFT dataset with diverse tasks.
    
    Args:
        total_count: Total number of examples
        deterministic_ratio: Fraction of deterministic tasks (for eval)
        seed: Random seed
    
    Returns:
        List of {"instruction": str, "response": str} dicts
    """
    random.seed(seed)
    
    num_deterministic = int(total_count * deterministic_ratio)
    num_other = total_count - num_deterministic
    
    # Generate examples
    examples = []
    
    # Deterministic tasks
    examples.extend(generate_deterministic_examples(num_deterministic))
    
    # Distribute rest across categories
    num_qa = int(num_other * 0.4)
    num_rewrite = int(num_other * 0.3)
    num_reason = num_other - num_qa - num_rewrite
    
    examples.extend(generate_qa_examples(num_qa))
    examples.extend(generate_rewriting_examples(num_rewrite))
    examples.extend(generate_reasoning_examples(num_reason))
    
    # Shuffle
    random.shuffle(examples)
    
    return examples[:total_count]


def save_to_jsonl(examples: List[Dict], output_path: str):
    """Save examples to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_path}")


def create_sft_splits(
    train_size: int = 2000,
    val_size: int = 200,
    test_size: int = 200,
    output_dir: str = "data",
    seed: int = 42,
):
    """
    Create train/val/test splits for SFT.
    
    Args:
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples
        output_dir: Output directory
        seed: Random seed
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate each split separately to ensure correct sizes
    random.seed(seed)
    
    train_examples = generate_synthetic_sft_data(train_size, seed=seed)
    val_examples = generate_synthetic_sft_data(val_size, seed=seed+1)
    test_examples = generate_synthetic_sft_data(test_size, seed=seed+2)
    
    # Save
    save_to_jsonl(train_examples, str(output_path / "sft_train.jsonl"))
    save_to_jsonl(val_examples, str(output_path / "sft_val.jsonl"))
    save_to_jsonl(test_examples, str(output_path / "sft_test.jsonl"))
    
    print(f"\n✓ Created SFT splits in {output_dir}/")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val:   {len(val_examples)}")
    print(f"  Test:  {len(test_examples)}")
