"""
Data augmentation for NeoBERT training.
Generates synthetic variants of benign packages to balance the dataset.
"""

import random
import re
import base64
from typing import List
from copy import deepcopy

from .unit_processor import PackageUnit


class CodeAugmenter:
    """Augment code samples to create synthetic training data."""

    def __init__(self, augmentation_prob: float = 0.3):
        self.augmentation_prob = augmentation_prob

    def augment_unit(self, unit: PackageUnit, augmentation_types: List[str]) -> List[PackageUnit]:
        """
        Apply augmentations to a single unit.

        Returns list of augmented units (including original).
        """
        augmented = [unit]  # Always include original

        for aug_type in augmentation_types:
            if random.random() < self.augmentation_prob:
                if aug_type == "minification":
                    aug_unit = self._minify(unit)
                elif aug_type == "comment_removal":
                    aug_unit = self._remove_comments(unit)
                elif aug_type == "variable_renaming":
                    aug_unit = self._rename_variables(unit)
                elif aug_type == "whitespace_variation":
                    aug_unit = self._vary_whitespace(unit)
                elif aug_type == "string_concat":
                    aug_unit = self._split_strings(unit)
                else:
                    continue

                if aug_unit:
                    augmented.append(aug_unit)

        return augmented

    def _minify(self, unit: PackageUnit) -> PackageUnit:
        """Minify code by removing whitespace and newlines."""
        new_unit = deepcopy(unit)

        # Simple minification
        content = unit.raw_content

        # Remove comments
        content = re.sub(r'//.*?\n', '\n', content)  # Single-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # Multi-line comments
        content = re.sub(r'#.*?\n', '\n', content)  # Python comments

        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        new_unit.raw_content = content
        new_unit.unit_id = f"{unit.unit_id}_minified"
        new_unit.unit_name = f"{unit.unit_name}_minified"

        # Re-tokenize and truncate to 512
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokens = tokenizer.tokenize(content)

        # Truncate to max 512 tokens
        if len(tokens) > 512:
            tokens = tokens[:512]

        new_unit.tokens = tokens
        new_unit.token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return new_unit

    def _remove_comments(self, unit: PackageUnit) -> PackageUnit:
        """Remove all comments from code."""
        new_unit = deepcopy(unit)

        content = unit.raw_content

        # Remove JavaScript/TypeScript comments
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Remove Python comments
        content = re.sub(r'#.*?\n', '\n', content)
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)

        new_unit.raw_content = content
        new_unit.unit_id = f"{unit.unit_id}_no_comments"
        new_unit.unit_name = f"{unit.unit_name}_no_comments"

        # Re-tokenize and truncate to 512
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokens = tokenizer.tokenize(content)

        # Truncate to max 512 tokens
        if len(tokens) > 512:
            tokens = tokens[:512]

        new_unit.tokens = tokens
        new_unit.token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return new_unit

    def _rename_variables(self, unit: PackageUnit) -> PackageUnit:
        """Rename variables to generic names (simple version)."""
        new_unit = deepcopy(unit)

        content = unit.raw_content

        # Common variable patterns
        var_pattern = r'\b([a-z_][a-z0-9_]{2,})\b'

        # Find all variable-like tokens
        variables = set(re.findall(var_pattern, content, re.IGNORECASE))

        # Reserved keywords to avoid
        reserved = {
            'function', 'const', 'let', 'var', 'if', 'else', 'for', 'while',
            'return', 'import', 'export', 'class', 'def', 'async', 'await',
            'from', 'as', 'in', 'of', 'True', 'False', 'None', 'null', 'undefined'
        }

        # Rename up to 10 variables
        renamed_count = 0
        for var in list(variables)[:10]:
            if var.lower() not in reserved and len(var) > 3:
                new_name = f"var{renamed_count}"
                content = re.sub(rf'\b{re.escape(var)}\b', new_name, content)
                renamed_count += 1

        new_unit.raw_content = content
        new_unit.unit_id = f"{unit.unit_id}_renamed"
        new_unit.unit_name = f"{unit.unit_name}_renamed"

        # Re-tokenize and truncate to 512
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokens = tokenizer.tokenize(content)

        # Truncate to max 512 tokens
        if len(tokens) > 512:
            tokens = tokens[:512]

        new_unit.tokens = tokens
        new_unit.token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return new_unit

    def _vary_whitespace(self, unit: PackageUnit) -> PackageUnit:
        """Vary whitespace (add/remove indentation)."""
        new_unit = deepcopy(unit)

        content = unit.raw_content
        lines = content.split('\n')

        # Randomly adjust indentation
        new_lines = []
        for line in lines:
            if line.strip():
                # Add or remove spaces
                if random.random() < 0.3:
                    extra_spaces = random.randint(0, 4)
                    line = ' ' * extra_spaces + line
            new_lines.append(line)

        content = '\n'.join(new_lines)

        new_unit.raw_content = content
        new_unit.unit_id = f"{unit.unit_id}_whitespace"
        new_unit.unit_name = f"{unit.unit_name}_whitespace"

        # Re-tokenize and truncate to 512
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokens = tokenizer.tokenize(content)

        # Truncate to max 512 tokens
        if len(tokens) > 512:
            tokens = tokens[:512]

        new_unit.tokens = tokens
        new_unit.token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return new_unit

    def _split_strings(self, unit: PackageUnit) -> PackageUnit:
        """Split string literals into concatenations."""
        new_unit = deepcopy(unit)

        content = unit.raw_content

        # Find string literals and split them
        def split_string(match):
            s = match.group(0)
            if len(s) > 10:  # Only split longer strings
                mid = len(s) // 2
                # JavaScript/Python string concatenation
                if "'" in s:
                    return f"{s[:mid]}' + '{s[mid:]}"
                else:
                    return f'{s[:mid]}" + "{s[mid:]}'
            return s

        # Split strings
        content = re.sub(r'"[^"]{10,}"', split_string, content)
        content = re.sub(r"'[^']{10,}'", split_string, content)

        new_unit.raw_content = content
        new_unit.unit_id = f"{unit.unit_id}_splitstr"
        new_unit.unit_name = f"{unit.unit_name}_splitstr"

        # Re-tokenize and truncate to 512
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokens = tokenizer.tokenize(content)

        # Truncate to max 512 tokens
        if len(tokens) > 512:
            tokens = tokens[:512]

        new_unit.tokens = tokens
        new_unit.token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return new_unit


def augment_benign_samples(benign_units: List[PackageUnit],
                           target_count: int,
                           augmentation_types: List[str]) -> List[PackageUnit]:
    """
    Augment benign samples to reach target count.

    Args:
        benign_units: Original benign units
        target_count: Desired number of benign units
        augmentation_types: Types of augmentation to apply

    Returns:
        Augmented list of benign units
    """
    augmenter = CodeAugmenter(augmentation_prob=0.5)

    all_units = list(benign_units)  # Start with originals

    while len(all_units) < target_count:
        # Randomly select a benign unit to augment
        original = random.choice(benign_units)

        # Apply random augmentation
        aug_type = random.choice(augmentation_types)
        augmented = augmenter.augment_unit(original, [aug_type])

        # Add augmented versions (skip original as it's already included)
        all_units.extend(augmented[1:])

    # Return exactly target_count units
    return all_units[:target_count]