"""
imports/quran_phonetizer_adapter.py
═══════════════════════════════════
Thin wrapper around the vendored ``imports.quran_transcript`` package that
retypes its outputs to plain dataclasses / JSON-friendly forms before
they cross into the rest of our codebase.

Why an adapter (instead of touching the vendored source in-place)
─────────────────────────────────────────────────────────────────
Modifying the vendored code in place would create merge friction if we
ever pull updates from upstream. The adapter pattern keeps the vendored
tree pristine; all retyping happens at this one boundary.

What's exposed
──────────────
- ``DEFAULT_MOSHAF_ATTRS``       : a sensible standard-Hafs preset for
                                    cases where the caller doesn't supply
                                    its own MoshafAttributes.
- ``phonetize_ayah(uthmani, m)`` : ergonomic front-door that returns
                                    ``(phonemes, mappings, ref_sifa_snapshots)``
                                    instead of the package's
                                    ``QuranPhoneticScriptOutput``.
- ``snapshot_ref_sifa(sifa_out)``: converts ONE ``SifaOutput`` (pydantic
                                    in the package) → one ``SifaSnapshot``
                                    (our plain dataclass).
- Re-exports the few package symbols we still need by name:
  ``Aya``, ``MoshafAttributes``, ``explain_error``, ``ReciterError``,
  ``MappingPos``, ``chunck_phonemes``.

  These are passed through unchanged because the downstream code that
  uses them already speaks the package's vocabulary (e.g.
  ``PhonemeError.from_reciter_error`` reads ``ReciterError`` attributes
  directly — see ``recitation_types.py``).
"""
from __future__ import annotations

from typing import Any, Optional

# Re-exports from the vendored package
from imports.quran_transcript import (
    Aya,
    MoshafAttributes,
    chunck_phonemes,
    explain_error,
    quran_phonetizer as _qt_phonetizer,
    MappingPos,
    ReciterError,
)

# Our own JSON-friendly sifat type
from recitation_types import SifaSnapshot, SIFAT_ATTRIBUTES


# ─────────────────────────────────────────────────────────────────────────────
#  Standard-Hafs preset
# ─────────────────────────────────────────────────────────────────────────────
# Five fields are required by ``MoshafAttributes``. The values below match
# the most-recited Hafs ʿan ʿĀṣim transmission (طريق الشاطبية): 4-beat
# Madd Munfasil and Muttasil, 6-beat at waqf for the Muttasil, 4-beat
# ʿĀriḍ. All other fields take the package's own defaults.
DEFAULT_MOSHAF_ATTRS = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=6,
    madd_aared_len=4,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Snapshot helpers
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_ref_sifa(sifa_out: Any, phoneme_group: str) -> SifaSnapshot:
    """Convert a reference ``SifaOutput`` (pydantic) → ``SifaSnapshot``.

    The reference sifat are deterministic (no confidences), so the
    ``confidence`` dict is empty.
    """
    snap = SifaSnapshot(phoneme_group=phoneme_group)
    for attr in SIFAT_ATTRIBUTES:
        val = getattr(sifa_out, attr, None)
        setattr(snap, attr, val)
    return snap


def snapshot_predicted_sifa(sifa_obj: Any) -> SifaSnapshot:
    """Convert a model-predicted ``Sifa`` (with ``SingleUnit`` per attr) →
    ``SifaSnapshot`` with probabilities attached in ``.confidence``.
    """
    snap = SifaSnapshot(phoneme_group=getattr(sifa_obj, "phonemes_group", ""))
    for attr in SIFAT_ATTRIBUTES:
        unit = getattr(sifa_obj, attr, None)
        if unit is None:
            continue
        # SingleUnit has .text (the categorical label) and .prob (confidence)
        setattr(snap, attr, getattr(unit, "text", None))
        prob = getattr(unit, "prob", None)
        if prob is not None:
            snap.confidence[attr] = float(prob)
    return snap


# ─────────────────────────────────────────────────────────────────────────────
#  Phonetize an ayah
# ─────────────────────────────────────────────────────────────────────────────

def phonetize_ayah(
    uthmani_text: str,
    moshaf: Optional[MoshafAttributes] = None,
) -> tuple[str, list[MappingPos], list[SifaSnapshot]]:
    """Phonetize an ayah and return our normalized 3-tuple.

    Parameters
    ──────────
    uthmani_text : the ayah's Uthmani script (with diacritics).
    moshaf       : MoshafAttributes; falls back to ``DEFAULT_MOSHAF_ATTRS``.

    Returns
    ───────
    (phonemes, mappings, ref_sifat_snapshots)

    Where:
      - ``phonemes`` is the phonetic-script string,
      - ``mappings`` is the per-char Uthmani→phoneme position list,
      - ``ref_sifat_snapshots`` has one ``SifaSnapshot`` per chunked
         phoneme (use ``chunck_phonemes(phonemes)`` to align).

    Invariant
    ─────────
    ``len(ref_sifat_snapshots) == len(chunck_phonemes(phonemes))``.
    """
    m = moshaf or DEFAULT_MOSHAF_ATTRS
    out = _qt_phonetizer(uthmani_text, m)

    # The package returns SifaOutput objects aligned 1:1 with chunked
    # phonemes. We turn each into our SifaSnapshot, attaching the
    # corresponding phoneme group as its label.
    chunks = chunck_phonemes(out.phonemes)
    assert len(chunks) == len(out.sifat), (
        f"Phonetizer invariant broken: {len(chunks)} chunks vs "
        f"{len(out.sifat)} sifat outputs"
    )
    snapshots = [
        snapshot_ref_sifa(s, phoneme_group=c)
        for c, s in zip(chunks, out.sifat)
    ]
    return out.phonemes, list(out.mappings), snapshots


# Smoke-test stub so the module is self-validating on import in dev:
if __name__ == "__main__":  # pragma: no cover
    ut = Aya(112, 1).get().uthmani
    ph, maps, snaps = phonetize_ayah(ut)
    print(f"Ayah: {ut!r}")
    print(f"Phonemes: {ph!r}")
    print(f"Mappings: {len(maps)}, Snapshots: {len(snaps)}")
    for sn in snaps:
        print(f"  {sn.phoneme_group!r}: tafkheem={sn.tafkheem_or_taqeeq}")
