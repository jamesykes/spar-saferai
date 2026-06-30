"""Project configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "detailed_estimates_OC3_DDoS_sota_saturated_40_repeats_5_LLMs_1_expert.csv"
)
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_valid_rows.csv"
SANITY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"
FITTED_DISTRIBUTIONS_DIR = PROJECT_ROOT / "outputs" / "fitted_distributions"
SOTA_BETA_FITS_PATH = FITTED_DISTRIBUTIONS_DIR / "sota_beta_fits.csv"
FORWARD_MODEL_SMOKE_TEST_DIR = PROJECT_ROOT / "outputs" / "forward_model_smoke_tests"

QUARTILE_COLUMNS = (
    "percentile_25th",
    "percentile_50th",
    "percentile_75th",
)

EXPECTED_LLM_MODELS = (
    "claude-sonnet-4-6",
    "gpt-5-mini",
    "gemini-3-flash-preview",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-7",
)

SOTA_TASKS = (
    "Paddle",
    "Labyrinth Linguist",
)

SATURATED_TASKS = (
    "pytorchLightning",
    "Randsubware",
)

EXPECTED_TASKS = SOTA_TASKS + SATURATED_TASKS

CAPABILITY_LEVEL_BY_TASK = {
    **{task: "SOTA" for task in SOTA_TASKS},
    **{task: "saturated" for task in SATURATED_TASKS},
}

EXPECTED_MITRE_STEP_LABELS = (
    "T1595 - Reconnaissance: Active Scanning",
    "T1590 - Reconnaissance: Gather Victim Network Information",
    "T1583.005 - Resource Development: Acquire Botnet",
    "T1584.005 - Resource Development: Build/Compromise Botnet",
    "T1036 - Defense Evasion: Masquerading",
    "T1571 - Defense Evasion: Non-Standard Port",
    "TA0011 - Command-and-Control",
    "T1498.001 - Impact: Direct Network Flood",
    "T1498.002 - Impact: Reflection/Amplification Attack",
)
