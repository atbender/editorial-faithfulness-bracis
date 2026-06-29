# Paradigm implementations for editorial faithfulness experiments

from .base import (
    Paradigm,
    ParadigmConfig,
    MCQAProblem,
    ExperimentalCondition,
    TrialResult,
    BASE_INSTRUCTION,
    SYSTEM_PROMPT,
)

from .ethical_information_access import EthicalInformationAccessParadigm
from .authority_bias import AuthorityBiasParadigm
from .reframing_bias import ReframingBiasParadigm

__all__ = [
    # Base classes
    "Paradigm",
    "ParadigmConfig",
    "MCQAProblem",
    "ExperimentalCondition",
    "TrialResult",
    "BASE_INSTRUCTION",
    "SYSTEM_PROMPT",
    # Paradigms
    "EthicalInformationAccessParadigm",
    "AuthorityBiasParadigm",
    "ReframingBiasParadigm",
]

