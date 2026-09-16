from __future__ import annotations
import re
from typing import TYPE_CHECKING
from imports.alphabet import phonetic_groups as phg
from imports.muaalem_typing import Unit, SingleUnit, Sifa
if TYPE_CHECKING:
    from imports.modeling.multi_level_tokenizer import MultiLevelTokenizer
import logging

def chunck_phonemes(phonetic_script: str) -> list[str]:
    """Chunk phonemes into groups
    Example:
    Inpupt: قَاالَ
    Output:
    قَ
    اا
    لَ
    """
    core_group = "|".join([f"{c}+" for c in phg.core])
    return re.findall(f"((?:{core_group})[{phg.residuals}]?)", phonetic_script)


def format_sifat(
    level_to_units: dict[str, list[Unit]],
    chunked_phonemes_batch: list[list[str]],
    multi_level_tokenizer: MultiLevelTokenizer,
) -> list[list[Sifa]]:
    sifat_batch = []
    for seq_idx in range(len(chunked_phonemes_batch)):
        sifat = []
        for idx, ph_group in enumerate(chunked_phonemes_batch[seq_idx]):
            sifa_dict = {}
            for level in level_to_units:
                if level == "phonemes":
                    continue
                sifa_idx = idx
                if sifa_idx < len(level_to_units[level][seq_idx].ids):
                    label = int(level_to_units[level][seq_idx].ids[sifa_idx])
                    text = multi_level_tokenizer.sifat_to_en_vocab[level][label]
                    p = level_to_units[level][seq_idx].probs[sifa_idx]
                    sifa_dict[level] = SingleUnit(
                        text=text, prob=float(p), idx=int(label)
                    )
                else:
                    logging.info(
                        f"Sequence: `{seq_idx}` has short Level: {level} we will place it with `None`"
                    )
                    sifa_dict[level] = None
            sifat.append(
                Sifa(
                    phonemes_group=chunked_phonemes_batch[seq_idx][idx],
                    **sifa_dict,
                )
            )
        sifat_batch.append(sifat)
    return sifat_batch