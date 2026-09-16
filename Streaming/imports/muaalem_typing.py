from __future__ import annotations
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Literal, Optional, TypeAlias, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    import torch

# The class below was adjusted
@dataclass
class Unit:
    """
    text: str : the predicted sequence of phonemes 
    probs: list of probabilities of the predicted units
    ids: list of ids of the predicted units
    frames: list[int] : Added to keep track of frames corresponding to the phonemes as well
    """
    text: str
    probs: torch.FloatTensor | list[float]
    ids: torch.LongTensor | list[int]
    frames: list[int] = None

    def __post_init__(self):
        if self.frames is None:
            self.frames = []


@dataclass
class SingleUnit:
    """
    A dataclass representing the predicted phoneme sequence with:
        text (str): Concatenated string of all phonemes.
        probs (Union[torch.FloatTensor, list[float]]):
            Confidence probabilities for each predicted phoneme (1D tensor).
        ids (Union[torch.LongTensor, list[int]]) (1D tensor):
            Token IDs corresponding to each phoneme.

    """

    text: str
    prob: float
    idx: int


@dataclass
class Sifa:
    """
    following optional properties (each is a SingleUnit or None):
        - phonemes_group (str): the phonemes associated with the `sifa`
        - hams_or_jahr (SingleUnit): either `hams` or `jahr`
        - shidda_or_rakhawa (SingleUnit): either `shadeed`, `between`, or `rikhw`
        - tafkheem_or_taqeeq (SingleUnit): either `mofakham`, `moraqaq`, or `low_mofakham`
        - itbaq (SingleUnit): either `monfateh`, or `motbaq`
        - safeer (SingleUnit): either `safeer`, or `no_safeer`
        - qalqla (SingleUnit): eithr `moqalqal`, or `not_moqalqal`
        - tikraar (SingleUnit): either `mokarar` or `not_mokarar`
        - tafashie (SingleUnit): either `motafashie`, or `not_motafashie`
        - istitala (SingleUnit): either `mostateel`, or `not_mostateel`
        - ghonna (SingleUnit): either `maghnoon`, or `not_maghnoon`

    Each SingleUnit in Sifa properties contains:
        text (str): The feature's categorical label (e.g., "hams", "shidda").
        prob (float): Confidence probability for this feature.
        idx (int): Identifier for the feature class.

    """

    phonemes_group: str
    hams_or_jahr: SingleUnit | None
    shidda_or_rakhawa: SingleUnit | None
    tafkheem_or_taqeeq: SingleUnit | None
    itbaq: SingleUnit | None
    safeer: SingleUnit | None
    qalqla: SingleUnit | None
    tikraar: SingleUnit | None
    tafashie: SingleUnit | None
    istitala: SingleUnit | None
    ghonna: SingleUnit | None

class SifaOutput(BaseModel):
    phonemes: str
    hams_or_jahr: Literal["hams", "jahr"]
    shidda_or_rakhawa: Literal["shadeed", "between", "rikhw"]
    tafkheem_or_taqeeq: Literal["mofakham", "moraqaq", "low_mofakham"]
    itbaq: Literal["monfateh", "motbaq"]
    safeer: Literal["safeer", "no_safeer"]
    qalqla: Literal["moqalqal", "not_moqalqal"]
    tikraar: Literal["mokarar", "not_mokarar"]
    tafashie: Literal["motafashie", "not_motafashie"]
    istitala: Literal["mostateel", "not_mostateel"]
    ghonna: Literal["maghnoon", "not_maghnoon"]

@dataclass
class LangName:
    ar: str
    en: str

@dataclass
class TajweedRule(ABC):
    """
    to_overwrite_tajweed_rules: if a rule in the future occpy in the same span ignore the new rule in the `to_overwrite_tajweed_rules` and keep the old rule
    """

    name: LangName
    golden_len: int
    correctness_type: Literal["match", "count"]
    tag: Optional[str] | None = None
    available_tags: Optional[set] | None = None

    def __post_init__(self):
        if self.tag is not None and self.available_tags is not None:
            if self.tag not in self.available_tags:
                raise ValueError(
                    f"Invalid tag value: `{self.tag}`. Available ones are: `{self.available_tags}`"
                )

    def count(self, ref_text, pred_text) -> int:
        return 0

    def match(self, ref_text, pred_text) -> bool:
        return True

    @abstractmethod
    def is_ph_str_in(self, ph_str: str) -> bool:
        """Whether the phonetic script is assoicated with this Tajweed rule or not"""
        return True

    @abstractmethod
    def get_relvant_rule(self, ph_str: str) -> Optional["TajweedRule"]:
        """Returs a Tajweed rule that is assocaited with the input ph_str"""
        return self

@dataclass
class MappingPos:
    """Represents character position mappings in Quranic text transformations.

    This dataclass tracks the relationship between character positions in the original
    text and their corresponding positions after regex substitution operations in the
    Quran transcription system. It maintains position spans and associated tajweed rules
    that apply to those character ranges.

    Attributes:
        pos: Tuple of (start, end) positions in the transformed text. The start is
            inclusive and the end is exclusive (Python-style slice notation).
        tajweed_rules: List of TajweedRule objects that apply to this character span.
            None indicates no tajweed rules are associated with this mapping.
        deleted(bool): Wheter this location is deleted or not. If deleted pos[0] == pos[1]

    Example:
        >>> mapping = MappingPos(pos=(0, 3), tajweed_rules=[])
        >>> print(mapping.pos)
        (0, 3)
        >>> # Add a tajweed rule to this mapping
        >>> mapping.add_tajweed_rule(None)  # No rule added
        >>> mapping.add_tajweed_rule(None)  # Still no rules
    """

    pos: tuple[int, int]  # start, (pythonic exlusive end)
    tajweed_rules: list[TajweedRule] | None = None
    deleted: bool = False

    def add_tajweed_rule(
        self, new_tajweed_rules: TajweedRule | list[TajweedRule] | None
    ) -> None:
        """Add a tajweed rule to this mapping position.

        Appends the new tajweed rule to the existing list of rules if both the
        current rules list and the new rule are not None.

        Args:
            new_tajweed_rule: The TajweedRule to add, or None if no rule to add.

        Example:
            >>> mapping = MappingPos(pos=(0, 3), tajweed_rules=[])
            >>> # This will add the rule if tajweed_rules exists and rule is not None
            >>> # mapping.add_tajweed_rule(some_rule)
        """
        if not new_tajweed_rules:  # covers None and []
            return
        if self.tajweed_rules is None:
            self.tajweed_rules = []
        # if new_tajweed_rules is a single rule, make it a list
        if isinstance(new_tajweed_rules, TajweedRule):
            self.tajweed_rules.append(new_tajweed_rules)
        else:
            self.tajweed_rules.extend(new_tajweed_rules)

MappingListType: TypeAlias = list[MappingPos]

@dataclass
class QuranPhoneticScriptOutput:
    phonemes: str
    sifat: list[SifaOutput]
    mappings: MappingListType  # `None` for deletion
    # TODO: Add mappings with sifat

@dataclass
class MuaalemOutput:
    """
    text (str): The feature's categorical label (e.g., "hams", "shidda").
    prob (float): Confidence probability for this feature.
    idx (int): Identifier for the feature class.
    """

    phonemes: Unit
    sifat: list[Sifa]
