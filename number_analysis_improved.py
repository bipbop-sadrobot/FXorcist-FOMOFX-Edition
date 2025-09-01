import os
import re
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd

# ==============================
# Setup Logging
# ==============================
LOG_FILE = "number_analysis.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==============================
# Load and Clean Data
# ==============================
def _read_data_file(file_path):
    """Helper function to read data from CSV, JSON, or TXT files."""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if df is not None:
                return df.iloc[:, 0].astype(str).tolist()
            else:
                return None
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
            if df is not None:
                return df.iloc[:, 0].astype(str).tolist()
            else:
                return None
        elif file_path.endswith(".sh"):
            with open(file_path, "r") as f:
                content = f.read()
                # Extract potential numbers from the shell script (very basic)
                numbers = re.findall(r"\d+", content)
                return numbers
        else:
            return None
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        return None

def load_numbers_from_file(file_path):
    """Load phone numbers from a JSON/CSV/TXT file and clean them."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    numbers = _read_data_file(file_path)

    if numbers is None:
        logger.warning(f"Could not read file as CSV, JSON, or SH: {file_path}")
        return []

    return [clean_number(num) for num in numbers if num]

def clean_number(number):
    """Remove non-digit characters and ensure standard formatting."""
    return re.sub(r"\D", "", number)

# ==============================
# Pattern Detection
# ==============================
# Precompile regular expressions
REPEATED_DIGITS = re.compile(r"(\d)\1{2,}").search
MNEMONIC_APPEAL = re.compile(r"(69|420|1337|777)").search

def detect_patterns(number):
    """Detect repeating or desirable patterns in a phone number."""
    patterns = []

    # Repeated digits (avoid simple ABAB)
    if REPEATED_DIGITS(number):
        patterns.append("Repeated Digits")

    # Mnemonic / phonetic appeal (basic vowel substitution idea)
    if MNEMONIC_APPEAL(number):
        patterns.append("Mnemonic Appeal")

    # Palindrome-like
    if number == number[::-1]:
        patterns.append("Palindrome")

    return patterns

# ==============================
# Statistical Scoring
# ==============================
def score_number(number):
    """Score number based on rarity of patterns."""
    patterns = detect_patterns(number)
    score = 0

    weights = {
        "Repeated Digits": 2,
        "Mnemonic Appeal": 3,
        "Palindrome": 4,
    }

    for p in patterns:
        score += weights.get(p, 1)

    return score, patterns

# ==============================
# Analysis Runner
# ==============================
def analyze_numbers(numbers):
    """Analyze list of numbers and rank by score."""
    logger.debug(f"Analyzing numbers: {numbers}")
    df = pd.DataFrame({"Number": numbers})
    df[["Score", "Patterns"]] = df["Number"].apply(lambda x: pd.Series(score_number(x)))
    df["Patterns"] = df["Patterns"].apply(lambda x: ", ".join(x) if x else "None")
    df.sort_values(by="Score", ascending=False, inplace=True)
    return df

# ==============================
# Save Results
# ==============================
def save_results(df, output_file="analysis_results.csv"):
    """Save analysis results to CSV."""
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.exception(f"Failed to save results: {e}")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    input_file = "data/data/raw/data.sh"  # change as needed
    logger.info("Starting number analysis...")

    numbers = load_numbers_from_file(input_file)

    if not numbers:
        logger.warning("No numbers found. Exiting.")
        exit()

    logger.info(f"Loaded {len(numbers)} numbers for analysis.")

    results_df = analyze_numbers(numbers)
    save_results(results_df)

    logger.info("Analysis complete.")
