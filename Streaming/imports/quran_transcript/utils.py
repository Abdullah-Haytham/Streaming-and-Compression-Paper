from pathlib import Path
import json
from dataclasses import dataclass
import re
from typing import Optional
import warnings

import xmltodict

from . import alphabet as alpha

BASE_PATH = Path(__file__).parent


class PartOfUthmaniWord(Exception):
    pass


@dataclass
class RasmFormat:
    uthmani: list[list[str]]
    imlaey: list[list[str]]


@dataclass
class Vertex:
    aya_idx: int
    word_idx: int


@dataclass
class WordSpan:
    start: int
    end: int | None


@dataclass
class AyaFormat:
    sura_idx: int
    aya_idx: int
    sura_name: str
    num_ayat_in_sura: int
    uthmani: str
    uthmani_words: list[str]
    imlaey_words: list[str]
    imlaey: str
    istiaatha_uthmani: str
    istiaatha_imlaey: str
    rasm_map: dict[str, list[str]] | None = None
    bismillah_uthmani: str | None = None
    bismillah_imlaey: str | None = None
    bismillah_map: dict[str, list[str]] | None = None
    """
    Attributes:
        sura_idx (int): the absoulte index of the sura starting form 1
        aya_idx (int): the absoulte index of the aya starting from 1
        sura_name (str): the name of the sura
        num_aya_in_sura (int): number of ayat in the sura
        uthmani (str): the uthmani script of the aya
        imlaey (str): the imlaey script of the aya
        istiaatha_uthmani (str): the Istiaatha in Uthmani script
        istiaatha_imlaey (str): the Istiaatha in Imlaey script

        rasm_map (list[dict[str, str]]): maping from imaley to uthmani
            scritps (word of uthmani to word or words of imlaey) and the
            opesite. Example:
            rasm_map=[
                {'@uthmani': 'مِنَ', '@imlaey': 'مِنَ'},
                {'@uthmani': 'ٱلْجِنَّةِ', '@imlaey': 'الْجِنَّةِ'},
                {'@uthmani': 'وَٱلنَّاسِ', '@imlaey': 'وَالنَّاسِ'}]
            Every item in the item is a dict with "@uthmain" and
            if None: the rasem map is not set yet

        bismillah_uthmani (str): bismillah in uthmani script if the
            aya index == 1 and the sura has bismillah or bismillah is
            not aya like sura Alfateha and else (None)

        bismillah_imlaey (str): bismillah in uthmani script if the
            aya index == 1 and the sura has bismillah or bismillah is
            not aya like sura Alfateha and else (None)

        bismillah_map (list[dict[str, str]]): maping from imaley to uthmani
            scritps (word of uthmani to word or words of imlaey) and
            the opesite. Example:
            bismillah_map=[
                {'@uthmani': 'بِسْمِ', '@imlaey': 'بِسْمِ'},
                {'@uthmani': 'ٱللَّهِ', '@imlaey': 'اللَّهِ'},
                {'@uthmani': 'ٱلرَّحْمَـٰنِ', '@imlaey': 'الرَّحْمَٰنِ'},
                {'@uthmani': 'ٱلرَّحِيمِ', '@imlaey': 'الرَّحِيمِ'}]
                Every item in the item is a dict with "@uthmain" and
            if None: the aya is not the first aya of the sura
            (Note: bismillah maping is set automaticllay no by the user)
    """

    def get_formatted_rasm_map(
        self,
        join_prefix=" ",
        uthmani_key="@uthmani",
        imlaey_key="@imlaey",
    ) -> RasmFormat:
        """
        return rasm map in fromt like:
            [
                {'@uthmani: str, '@imlaey: str},
                {'@uthmani: str, '@imlaey: str},
            ]
            to
            RasmFormat.uthmani: list[list[str]]
            RasmFormat.imlaey: list[list[str]]
        """
        if self.rasm_map is None:
            raise ValueError("Rasmp map is None")

        uthmani_words: list[list[str]] = []
        imlaey_words: list[list[str]] = []
        for item in self.rasm_map:
            uthmani_words.append(item[uthmani_key].split(join_prefix))
            imlaey_words.append(item[imlaey_key].split(join_prefix))
        return RasmFormat(uthmani=uthmani_words, imlaey=imlaey_words)


@dataclass
class EncodingOutput:
    """
    Output container for Quranic text encoding operations.

    Attributes:
        imlaey2uthmani: Mapping from Imlaey word indices to Uthmani word indices
        uthmani_words: List of Uthmani script words
        imlaey_words: List of Imlaey script words
        aya_imlaey_span_words: Tuple (start, end) of Imlaey word indices for core Quranic content
        istiaatha_imlaey_span_words: Span for Istiaatha (أعوذ بالله) if present, else None
        bismillah_imlaey_span_words: Span for Bismillah (بسم الله) if present, else None
        sadaka_imlaey_span_words: Span for Sadaka (صدق الله) if present, else None

    Note:
        All spans use Python-style exclusive indexing:
        - start: inclusive index
        - end: exclusive index
        Example: (2, 4) covers words at indices 2 and 3
    """

    imlaey2uthmani: dict[int, int]
    uthmani_words: list[str]
    imlaey_words: list[str]
    aya_imlaey_span_words: tuple[int, int]
    istiaatha_imlaey_span_words: tuple[int, int] | None
    bismillah_imlaey_span_words: tuple[int, int] | None
    sadaka_imlaey_span_words: tuple[int, int] | None


@dataclass
class QuranWordIndex:
    """
    Represents bidirectional word indices between scripts using exclusive indexing.
    Can be used as a start or end

    Attributes:
        imlaey: Word index in Imlaey script (exclusive boundary)
        uthmani: Word index in Uthmani script (exclusive boundary)

    Example:
        Given words = ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَـٰنِ", "ٱلرَّحِيمِ"]
        start = QuranWordIndex(imlaey=1, uthmani=1) →
            Starts AFTER first word ("بِسْمِ")
        end = QuranWordIndex(imlaey=3, uthmani=3) →
            Position BEFORE third word ("ٱلرَّحْمَـٰنِ") in output
    """

    imlaey: int
    uthmani: int


@dataclass
class Imlaey2uthmaniOutput:
    """
    Container for Imlaey to Uthmani conversion results.

    Attributes:
        imlaey: Imlaey script text segment
        uthmani: Converted Uthmani script text
        quran_start: Starting Quran word indices (None if no Quran content)
        quran_end: Ending Quran word indices (None if no Quran content)
        has_istiaatha: True if segment contains Istiaatha (أعوذ بالله من الشيطان الرحيم)
        has_bismillah: True if segment contains Bismillah (بسم الله الرحمن الريحم)
        has_sadaka: True if segment contains Sadaka (صدق الله العظيم)
        has_quran: True if segment contains core Quranic text.
            Istiaath, Bismillah and Sadaka are not considered a part of Holy Quan

    Note:
        - quran_start and quran_end use exclusive indexing (Python-style)
        - Bismillah in Surah 1:1 and Surah 27:30 is considered Quranic content, so `has_bismillah` will be `False`
        - Non-Quran components: Istiaatha, Bismillah (when not part of verse), Sadaka
        Example:
            quran_start=QuranWordIndex(imlaey=0, uthmani=0) → Beginning of text
            quran_end=QuranWordIndex(imlaey=4, uthmani=4) → After fourth word
            position 4 is not counted in the output
    """

    imlaey: str
    uthmani: str
    quran_start: QuranWordIndex | None
    quran_end: QuranWordIndex | None
    has_istiaatha: bool
    has_bismillah: bool
    has_sadaka: bool
    has_quran: bool

    # NOTE: At suraht Alfatiha (1) in Aya (1) ans Surhat Alnaml (27) aya (3)
    # بسم الله الرحمن الرحيم is considerd an aya so `has_bismillah` will be `False`


@dataclass
class SegmentScripts:
    """
    Container for segmented Quranic text with dual indexing.

    Attributes:
        imalaey: Full Imlaey script text
        uthmani: Full Uthmani script text
        has_istiaatha: True if segment contains Istiaatha (أعوذ بالله من الشيطان الرجيم)
        has_bismillah: True if segment contains Bismillah  (بسما لله الرحمن الريحم)
        has_sadaka: True if segment contains Sadaka (صدق الله العظيم)
        has_quran: True if segment contains core Quranic text
        start_span: (sura_idx, aya_idx, word_index) for start position
        end_span: (sura_idx, aya_idx, word_index) for end position

    Note:
        - Indices: sura_idx (1-114), aya_idx (≥1)
        - Bismillah in Surah 1:1 and Surah 27:30 is considered Quranic aya, so `has_bismillah` will be `False`
        - Sura/aya indices use INCLUSIVE 1-based indexing (Quranic standard)
        - Word indices use EXCLUSIVE 0-based indexing (Python standard)
        - Spans are None if has_quran=False
        Example:
            (1, 1, QuranWordIndex(imlaey=0, uthmani=0)) →
                Surah 1, Ayah 1, starting at first word
            (2, 3, QuranWordIndex(imlaey=5, uthmani=5)) →
                Surah 2, Ayah 3, before 6th word (word index 5 is not included)
    """

    imalaey: str
    uthmani: str
    has_istiaatha: bool
    has_bismillah: bool
    has_sadaka: bool
    has_quran: bool
    start_span: tuple[int, int, QuranWordIndex] | None
    end_span: tuple[int, int, QuranWordIndex] | None

    # NOTE: At suraht Alfatiha (1) in Aya (1) ans Surhat Alnaml (27) aya (3)
    # بسم الله الرحمن الرحيم is considerd an aya so `has_bismillah` will be `False`
    """
    Attirubutes:
        start_span (tuple[int, int] | None): (start sura index from 1 to 114, start aya index from 1) or `None`
            if `has_quran` is `False`
        end_span (tuple[int, int] None): (end sura index from 1 to 114, end aya index from 1)
            Note: `sura_index` and `aya_index` are inclusive indexing  unlike python indexing and QuranWordIndex in execlusive
            just like python indexing
            or `None` if `has_quran` is `False`

        has_quran (bool): whether the segment part contains quran or not.
            Note: `bimillah` (بسم الله الحرم الريحم) , `istiaatha` (أعوذ بالله من الشيطان الرجيم)
            and `sadaka` (صدق اللع العظيم) are not considered part of the Holy Quran
    """


# TODO: Add quran_dict as default
class Aya(object):
    def __init__(
        self,
        sura_idx=1,
        aya_idx=1,
        quran_path: str | Path = BASE_PATH / "quran-script/quran-uthmani-imlaey.json",
        quran_dict: Optional[dict] = None,
        start_imlaey_word_idx: Optional[int] = None,
        prefix="@",
        map_key="rasm_map",
        bismillah_map_key="bismillah_map",
        bismillah_key="bismillah",
        uthmani_key="uthmani",
        imlaey_key="imlaey",
        sura_name_key="name",
        join_prefix=" ",
    ):
        """
        quran_path (str | Path) path to the quran json script with
            emlaey uthmani scripts
        sura_idx: the index of the Sura in the Quran starting with 1 to 114
        aya_idx: the index of the aya starting form 1
        """
        self.quran_path = Path(quran_path)
        if quran_dict is None:
            with open(self.quran_path, "r", encoding="utf8") as f:
                self.quran_dict = json.load(f)
        else:
            self.quran_dict = quran_dict

        # Loading Istiaath
        self.istiaatha_imlaey = alpha.istiaatha.imlaey
        self.istiaatha_uthmani = alpha.istiaatha.uthmani

        self._check_indices(sura_idx - 1, aya_idx - 1)
        # NOTE: we are storing sura index and aya index as absolute index (starting from 0 not 1)
        # TODO: confuse naming we should make it clean that is diffrent for user
        # exepctations we should name it python_sura_idx to diffrentiate it
        self.sura_idx = sura_idx - 1
        self.aya_idx = aya_idx - 1

        self.map_key = map_key
        self.bismillah_map_key = bismillah_map_key
        self.uthmani_key = prefix + uthmani_key
        self.imlaey_key = prefix + imlaey_key
        self.sura_name_key = prefix + sura_name_key
        self.bismillah_uthmani_key = f"{prefix}{bismillah_key}_{uthmani_key}"
        self.bismillah_imlaey_key = f"{prefix}{bismillah_key}_{imlaey_key}"
        self.join_prefix = join_prefix

        # NOTE: this variables used in by word steping for imlaey script in methods:
        # * ``
        # * ``
        if start_imlaey_word_idx is None:
            self.start_imlaey_word_idx = 0
        else:
            self.start_imlaey_word_idx = start_imlaey_word_idx
        self.decoding_cache = {}

    def get_start_imlaey_word_idx(self):
        return self.start_imlaey_word_idx

    def _get_sura(self, sura_idx):
        assert sura_idx >= 0 and sura_idx <= 113, f"Wrong Sura index {sura_idx + 1}"
        return self.quran_dict["quran"]["sura"][sura_idx]["aya"]

    def _get_sura_object(self, sura_idx):
        assert sura_idx >= 0 and sura_idx <= 113, f"Wrong Sura index {sura_idx + 1}"
        return self.quran_dict["quran"]["sura"][sura_idx]

    def _get_aya(self, sura_idx, aya_idx):
        assert aya_idx >= 0 and aya_idx < len(self._get_sura(sura_idx)), (
            f"Sura index out of range sura_index={sura_idx + 1} "
            + f"and len of sura={len(self._get_sura(sura_idx))}"
        )
        return self._get_sura(sura_idx)[aya_idx]

    def _get(self, sura_idx, aya_idx) -> AyaFormat:
        """
        get an aya from quran script
        Args:
            sura_idx (int): from 0 to 113
            aya_idx (int): form 0 to len(sura) - 1
        Example to get the first aya of sura Alfateha quran_scirt[1, 1]
        Return:
            AyaFormt:
                sura_idx (int): the absoulte index of the sura
                aya_idx (int): the absoulte index of the aya
                sura_name (str): the name of the sura
                num_aya_in_sura (int): number of ayat in the sura
                uthmani (str): the uthmani script of the aya
                imlaey (str): the imlaey script of the aya
                istiaatha_uthmani (str): the Istiaatha in Uthmani script
                istiaatha_imlaey (str): the Istiaatha in Imlaey script

                rasm_map (list[dict[str, str]]): maping from imaley to uthmani
                    scritps (word of uthmani to word or words of imlaey) and the
                    opesite. Example:
                    rasm_map=[
                        {'@uthmani': 'مِنَ', '@imlaey': 'مِنَ'},
                        {'@uthmani': 'ٱلْجِنَّةِ', '@imlaey': 'الْجِنَّةِ'},
                        {'@uthmani': 'وَٱلنَّاسِ', '@imlaey': 'وَالنَّاسِ'}]
                    Every item in the item is a dict with "@uthmain" and
                    if None: the rasem map is not set yet

                bismillah_uthmani (str): bismillah in uthmani script if the
                    aya index == 1 and the sura has bismillah or bismillah is
                    not aya like sura Alfateha and else (None)

                bismillah_imlaey (str): bismillah in uthmani script if the
                    aya index == 1 and the sura has bismillah or bismillah is
                    not aya like sura Alfateha and else (None)

                bismillah_map (list[dict[str, str]]): maping from imaley to uthmani
                    scritps (word of uthmani to word or words of imlaey) and
                    the opesite. Example:
                    bismillah_map=[
                        {'@uthmani': 'بِسْمِ', '@imlaey': 'بِسْمِ'},
                        {'@uthmani': 'ٱللَّهِ', '@imlaey': 'اللَّهِ'},
                        {'@uthmani': 'ٱلرَّحْمَـٰنِ', '@imlaey': 'الرَّحْمَٰنِ'},
                        {'@uthmani': 'ٱلرَّحِيمِ', '@imlaey': 'الرَّحِيمِ'}]
                        Every item in the item is a dict with "@uthmain" and
                    if None: the aya is not the first aya of the sura
                    (Note: bismillah maping is set automaticllay no by the user)
        """
        bismillah = {self.bismillah_uthmani_key: None, self.bismillah_imlaey_key: None}
        for key in bismillah.keys():
            if key in self._get_aya(sura_idx, aya_idx).keys():
                bismillah[key] = self._get_aya(sura_idx, aya_idx)[key]

        bismillah_map = None
        if self.bismillah_map_key in self._get_aya(sura_idx, aya_idx).keys():
            bismillah_map = self._get_aya(sura_idx, aya_idx)[self.bismillah_map_key]

        rasm_map = None
        if self.map_key in self._get_aya(sura_idx, aya_idx).keys():
            rasm_map = self._get_aya(sura_idx, aya_idx)[self.map_key]

        return AyaFormat(
            sura_idx=sura_idx + 1,
            aya_idx=aya_idx + 1,
            sura_name=self._get_sura_object(sura_idx)[self.sura_name_key],
            num_ayat_in_sura=len(self._get_sura(sura_idx)),
            uthmani=self._get_aya(sura_idx, aya_idx)[self.uthmani_key],
            uthmani_words=self._get_aya(sura_idx, aya_idx)[self.uthmani_key].split(
                self.join_prefix
            ),
            imlaey=self._get_aya(sura_idx, aya_idx)[self.imlaey_key],
            imlaey_words=self._get_aya(sura_idx, aya_idx)[self.imlaey_key].split(
                self.join_prefix
            ),
            rasm_map=rasm_map,
            bismillah_uthmani=bismillah[self.bismillah_uthmani_key],
            bismillah_imlaey=bismillah[self.bismillah_imlaey_key],
            bismillah_map=bismillah_map,
            istiaatha_uthmani=self.istiaatha_uthmani,
            istiaatha_imlaey=self.istiaatha_imlaey,
        )

    def get(self) -> AyaFormat:
        """
        get an aya from quran script
        Return:
            AyaFormt:
                sura_idx (int): the absoulte index of the sura
                aya_idx (int): the absoulte index of the aya
                sura_name (str): the name of the sura
                uthmani (str): the uthmani script of the aya
                imlaey (str): the imlaey script of the aya

                bismillah_uthmani (str): bismillah in uthmani script if the
                    aya index == 1 and the sura has bismillah or bismillah is
                    not aya like sura Alfateha and else (None)

                bismillah_imlaey (str): bismillah in uthmani script if the
                    aya index == 1 and the sura has bismillah or bismillah is
                    not aya like sura Alfateha and else (None)
        """

        return self._get(self.sura_idx, self.aya_idx)

    def is_last(self) -> bool:
        """Whether the aya is the last aya in the sura or not"""
        return (self.aya_idx + 1) == self.get().num_ayat_in_sura

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return str(self.get())

    def _check_indices(self, sura_idx: int, aya_idx: int):
        """
        check sura ds compatibility
        """
        assert sura_idx >= 0 and sura_idx <= 113, f"Wrong Sura index {sura_idx + 1}"

        assert aya_idx >= 0 and aya_idx < len(self._get_sura(sura_idx)), (
            f"Aya index out of range (sura_index={sura_idx + 1} "
            + f"aya_index={aya_idx + 1}) "
            + f"and length of sura={len(self._get_sura(sura_idx))}"
        )

    def _set_ids(self, sura_idx, aya_idx):
        self.sura_idx = sura_idx
        self.aya_idx = aya_idx

    def set(self, sura_idx, aya_idx, start_imlaey_word_idx: int | None = None) -> None:
        """Set the aya
        Args:
        sura_idx: the index of the Sura in the Quran starting with 1 to 114
        aya_idx: the index of the aya starting form 1
        """
        self._check_indices(sura_idx - 1, aya_idx - 1)
        self._set_ids(sura_idx=sura_idx - 1, aya_idx=aya_idx - 1)
        if start_imlaey_word_idx:
            self.start_imlaey_word_idx = start_imlaey_word_idx

    def set_new(self, sura_idx, aya_idx, start_imlaey_word_idx: int | None = None):
        """Return new aya with sura, and aya indices
        Args:
        sura_idx: the index of the Sura in the Quran starting with 1 to 114
        aya_idx: the index of the aya starting form 1
        """
        return Aya(
            quran_path=self.quran_path,
            sura_idx=sura_idx,
            aya_idx=aya_idx,
            quran_dict=self.quran_dict,
            start_imlaey_word_idx=start_imlaey_word_idx,
        )

    def step(self, step_len: int) -> "Aya":
        """
        Return new Aya object with "step_len" aya after of before
        circular loop
        """
        aya_relative_idx = step_len + self.aya_idx

        # +VE or zero aya idx
        if aya_relative_idx >= 0:
            sura_idx = self.sura_idx
            while True:
                num_ayat = self._get(sura_idx=sura_idx, aya_idx=0).num_ayat_in_sura
                if aya_relative_idx < num_ayat:
                    break
                aya_relative_idx -= num_ayat
                sura_idx = (sura_idx + 1) % 114

        # -VE aya idx
        else:
            sura_idx = (self.sura_idx - 1) % 114
            while True:
                num_ayat = self._get(sura_idx=sura_idx, aya_idx=0).num_ayat_in_sura
                aya_relative_idx += num_ayat
                if aya_relative_idx >= 0:
                    break

        return Aya(
            quran_path=self.quran_path,
            sura_idx=sura_idx + 1,
            aya_idx=aya_relative_idx + 1,
            quran_dict=self.quran_dict,
        )

    # TODO: Add vertix
    def get_ayat_after(self, end_vertix=(114, 6), num_ayat: int | None = None):
        """
        iterator looping over Quran ayayt (verses) starting from the
        current aya to the end of the Holy Quran
        Args:
            num_aya: loop for ayat until reaching aya + num_ayat - 1
        """
        if num_ayat is not None:
            aya = self
            for _ in range(num_ayat):
                yield aya
                aya = aya.step(1)
            return

        # TODO: subject to end_vertix
        aya_start_idx = self.aya_idx
        for sura_loop_idx in range(self.sura_idx, 114):
            for aya_loop_idx in range(
                aya_start_idx, len(self._get_sura(sura_loop_idx))
            ):
                yield Aya(
                    quran_path=self.quran_path,
                    sura_idx=sura_loop_idx + 1,
                    aya_idx=aya_loop_idx + 1,
                    quran_dict=self.quran_dict,
                )
            aya_start_idx = 0

    def _get_map_dict(
        self, uthmani_list: list[str], imlaey_list: list[str]
    ) -> list[dict[str, str]]:
        """
        Return:
            [
                {'@uthmani: str, '@imlaey: str},
                {'@uthmani: str, '@imlaey: str},
            ]
        """
        map_list: list[str] = []
        for uthmani_words, imlaey_words in zip(uthmani_list, imlaey_list):
            map_list.append(
                {
                    self.uthmani_key: self.join_prefix.join(uthmani_words),
                    self.imlaey_key: self.join_prefix.join(imlaey_words),
                }
            )
        return map_list

    def _get_str_from_lists(self, L: list[list[str]]) -> str:
        """
        join a list of lists of str with (self.join_prefix)
        Example: :
            L = [
                    ['a', 'b'],
                    ['c', 'd', 'e']
                ]
            self.join_prefic = ' '
            Ouput: 'a b c d e'
        """
        return self.join_prefix.join([self.join_prefix.join(x) for x in L])

    def set_rasm_map(
        self,
        uthmani_list: list[list[str]],
        imlaey_list: list[list[str]],
    ):
        # Assert len
        assert len(uthmani_list) == len(imlaey_list), (
            f"Lenght mismatch: len(uthmani)={len(uthmani_list)} "
            + f"and len(imlaey)={len(imlaey_list)}"
        )

        # assert missing script
        # (Uthmani)
        assert self._get_str_from_lists(uthmani_list) == self.get().uthmani, (
            f"Missing Uthmani script words! input_uthmani_list={uthmani_list}"
            + f"\nAnd the original uthmani Aya={self.get().uthmani}"
        )
        # (Imlaey)
        assert self._get_str_from_lists(imlaey_list) == self.get().imlaey, (
            f"Missing Imlaey script words! input_imlaey_list={imlaey_list}"
            + f"\nAnd the original imlaey Aya={self.get().imlaey}"
        )

        # check first aya (set bismillah map)
        bismillah_map = None
        if (
            self.get().bismillah_uthmani is not None
            and self.get().bismillah_map is None
        ):
            bismillah_uthmani = self.get().bismillah_uthmani.split(self.join_prefix)
            bismillah_uthmani = [[word] for word in bismillah_uthmani]
            bismillah_imlaey = self.get().bismillah_imlaey.split(self.join_prefix)
            bismillah_imlaey = [[word] for word in bismillah_imlaey]

            bismillah_map = self._get_map_dict(
                uthmani_list=bismillah_uthmani, imlaey_list=bismillah_imlaey
            )

        # get rasm map
        rasm_map = self._get_map_dict(
            uthmani_list=uthmani_list, imlaey_list=imlaey_list
        )

        # save quran script file
        self.quran_dict["quran"]["sura"][self.sura_idx]["aya"][self.aya_idx][
            self.map_key
        ] = rasm_map
        if bismillah_map is not None:
            self.quran_dict["quran"]["sura"][self.sura_idx]["aya"][self.aya_idx][
                self.bismillah_map_key
            ] = bismillah_map

    def save_quran_dict(self):
        # save the file
        with open(self.quran_path, "w+", encoding="utf8") as f:
            json.dump(self.quran_dict, f, ensure_ascii=False, indent=2)

        # # TODO for debuging
        # with open(self.quran_path.parent / 'text.xml', 'w+', encoding='utf8') as f:
        #     new_file = xmltodict.unparse(self.quran_dict, pretty=True)
        #     f.write(new_file)

    def get_formatted_rasm_map(self) -> RasmFormat:
        """
        return rasm map in fromt like:
            [
                {'@uthmani: str, '@imlaey: str},
                {'@uthmani: str, '@imlaey: str},
            ]
            to
            RasmFormat.uthmani: list[list[str]]
            RasmFormat.imlaey: list[list[str]]
        """
        if self.get().rasm_map is None:
            raise ValueError("Rasmp map is None")

        uthmani_words: list[list[str]] = []
        imlaey_words: list[list[str]] = []
        for item in self.get().rasm_map:
            uthmani_words.append(item[self.uthmani_key].split(self.join_prefix))
            imlaey_words.append(item[self.imlaey_key].split(self.join_prefix))
        return RasmFormat(uthmani=uthmani_words, imlaey=imlaey_words)

    def _encode_imlaey_to_uthmani(
        self,
        include_bismillah=False,
        include_istiaatha=False,
        include_sadaka=False,
    ) -> EncodingOutput:
        """
        Encodes Imlaey text into Uthmani script with optional prefixes/suffixes.

        Args:
            include_bismillah (bool): If True, includes "Bismillah" (بسم الله الرحمن الرحيم)
                as part of the first ayah's encoding. Note: Bismillah is automatically
                included in Surah Al-Fatihah (as it is considered an ayah) and excluded
                in Surah At-Tawbah (no Bismillah in this surah).

            include_istiaatha (bool): If True, includes the Istiaatha (أعوذ بالله من الشيطان الرجيم)
                at the beginning of the surah (only for the first ayah).

            include_sadaka (bool): If True, appends "Sadaka Allahu Al-'Azeem" (صدق الله العظيم)
                after the last ayah of the surah.


        Notes:
            - Warnings are issued if Istiaatha, Bismillah, or Sadaka are requested in invalid positions.
            - Handles edge cases where Uthmani and Imlaey word counts differ (e.g., due to unique Rasm rules).
        """
        # caching decoding (indexing starge) for fasster infernce
        if self.decoding_cache:
            if (
                self.decoding_cache["include_istiaatha"] == include_istiaatha
                and self.decoding_cache["include_bismillah"] == include_bismillah
                and self.decoding_cache["include_sadaka"] == include_sadaka
            ):
                imlaey2uthmani = self.decoding_cache["imlaey2uthmani"]
                uthmani_words = self.decoding_cache["uthmani_words"]
                imlaey_words = self.decoding_cache["imlaey_words"]
                return EncodingOutput(
                    imlaey2uthmani=imlaey2uthmani,
                    uthmani_words=uthmani_words,
                    imlaey_words=imlaey_words,
                    aya_imlaey_span_words=self.decoding_cache["aya_imlaey_span_words"],
                    istiaatha_imlaey_span_words=self.decoding_cache[
                        "istiaatha_imlaey_span_words"
                    ],
                    bismillah_imlaey_span_words=self.decoding_cache[
                        "bismillah_imlaey_span_words"
                    ],
                    sadaka_imlaey_span_words=self.decoding_cache[
                        "sadaka_imlaey_span_words"
                    ],
                )

        uthmani_words = []
        imlaey_words = []
        istiaatha_imlaey_span_words = None
        bismillah_imlaey_span_words = None
        sadaka_imlaey_span_words = None
        # NOTE: include istiaathta only at the begining of the sura
        if include_istiaatha:
            if (self.aya_idx + 1) == 1:
                ist_start = len(imlaey_words)
                uthmani_words += alpha.istiaatha.uthmani.split(self.join_prefix)
                imlaey_words += alpha.istiaatha.imlaey.split(self.join_prefix)
                istiaatha_imlaey_span_words = (ist_start, len(imlaey_words))
            else:
                warnings.warn(
                    f"Istiaatha will not be included. We only include Istiaatha at the beginning of every sura (first aya only). Aya index is: `{self.aya_idx + 1}`"
                )
        # NOTE: we inlcude bimillah only at the first of every sura except for every sura number 9
        # surah Al-Tawba
        # bismillah is part of surah Al fatiha so according to Hafs so we do not
        # inlcude it as it is already an aya
        if include_bismillah:
            if self.get().bismillah_uthmani is not None:
                bis_start = len(imlaey_words)
                uthmani_words += self.get().bismillah_uthmani.split(self.join_prefix)
                imlaey_words += self.get().bismillah_imlaey.split(self.join_prefix)
                bismillah_imlaey_span_words = (bis_start, len(imlaey_words))
            else:
                warnings.warn(
                    f"Bismillah will not be included, as it is only placed at the beginning of each surah (except Surah At-Tawbah (9)). Note: Bismillah is counted as an ayah in Surah Al-Fatiha (1). The sura is : `{self.sura_idx + 1}` and Aya is: `{self.aya_idx + 1}`"
                )
        # The Aya itself
        aya_imlaey_span_start = len(imlaey_words)
        uthmani_words += self.get().uthmani_words
        imlaey_words += self.get().imlaey_words
        aya_imlaey_span_words = (aya_imlaey_span_start, len(imlaey_words))

        # NOTE: include sadaka and the aya is the last aya in the sura only
        if include_sadaka:
            if (self.aya_idx + 1) == self.get().num_ayat_in_sura:
                s_start = len(imlaey_words)
                uthmani_words += alpha.sadaka.uthmani.split(self.join_prefix)
                imlaey_words += alpha.sadaka.imlaey.split(self.join_prefix)
                sadaka_imlaey_span_words = (s_start, len(imlaey_words))
            else:
                warnings.warn(
                    f"صدق الله العظيم will not be included. We only include `sadaka` after the end of every sura. The Sura idx is: `{self.sura_idx + 1}`, the aya is: `{self.aya_idx + 1}` and the last aya is `{self.get().num_ayat_in_sura}` "
                )

        # Same words map to each other for both imlaey and uthmani
        if len(uthmani_words) == len(imlaey_words):
            imlaey2uthmani = {idx: idx for idx in range(len(uthmani_words))}

        else:
            # len mismatch: some words in uthmani map to more than words in the imlaey
            # for example: يبتنؤم in uthmani maps to يا ابن أم in imlaey
            iml_idx = 0
            imlaey2uthmani = {}
            for uth_idx in range(len(uthmani_words)):
                # special words of Uthmani Rasm
                span = self._get_unique_rasm_map_span(iml_idx, imlaey_words)
                if span is not None:
                    for idx in range(iml_idx, iml_idx + span):
                        imlaey2uthmani[idx] = uth_idx
                    iml_idx += span

                # words in uthmnai starts with يأيهاو هأنتم maps to two imlaey words
                # يا أيها ها أنتم
                elif imlaey_words[iml_idx] in alpha.unique_rasm.imlaey_starts:
                    imlaey2uthmani[iml_idx] = uth_idx
                    imlaey2uthmani[iml_idx + 1] = uth_idx
                    iml_idx += 2

                else:
                    imlaey2uthmani[iml_idx] = uth_idx
                    iml_idx += 1

        assert sorted(imlaey2uthmani.keys())[-1] == len(imlaey_words) - 1
        #
        assert sorted(imlaey2uthmani.values())[-1] == len(uthmani_words) - 1

        # Saving the claculated inices in cache
        self.decoding_cache["imlaey2uthmani"] = imlaey2uthmani
        self.decoding_cache["uthmani_words"] = uthmani_words
        self.decoding_cache["imlaey_words"] = imlaey_words
        self.decoding_cache["aya_imlaey_span_words"] = aya_imlaey_span_words
        self.decoding_cache["istiaatha_imlaey_span_words"] = istiaatha_imlaey_span_words
        self.decoding_cache["bismillah_imlaey_span_words"] = bismillah_imlaey_span_words
        self.decoding_cache["sadaka_imlaey_span_words"] = sadaka_imlaey_span_words
        self.decoding_cache["include_istiaatha"] = include_istiaatha
        self.decoding_cache["include_bismillah"] = include_bismillah
        self.decoding_cache["include_sadaka"] = include_sadaka

        return EncodingOutput(
            imlaey2uthmani=imlaey2uthmani,
            uthmani_words=uthmani_words,
            imlaey_words=imlaey_words,
            aya_imlaey_span_words=aya_imlaey_span_words,
            istiaatha_imlaey_span_words=istiaatha_imlaey_span_words,
            bismillah_imlaey_span_words=bismillah_imlaey_span_words,
            sadaka_imlaey_span_words=sadaka_imlaey_span_words,
        )

    def _get_unique_rasm_map_span(self, idx: int, words: list[int]) -> int | None:
        """
        check that words starting of idx is in alphabet.unique_rasm.rasm_map
        if that applies, it will return the number of imlaey words in
        alphabet.unique_rasm.rasm_map
        Else: None
        """
        for unique_rasm in alpha.unique_rasm.rasm_map:
            span = len(unique_rasm["imlaey"].split(self.join_prefix))
            if self.join_prefix.join(words[idx : idx + span]) == unique_rasm["imlaey"]:
                return span
        return None

    def _decode_uthmani(
        self,
        imlaey_wordspan: WordSpan,
        imlaey2uthmani: dict[int, int],
        uthmani_words: list[str],
    ) -> str:
        """
        Args:
            Imlaey_wordspan: (start, end):
                start: the start word idx in imlaey script of the aya
                end: the (end + 1) word idx in imlaey script of the aya if end
                    is None then means to the last word idx of the imlaey aya
        return the uthmani script of the given imlaey_word_span in
        Imlaey script Aya
        """
        start = imlaey_wordspan.start
        if imlaey_wordspan.end is None:
            end = len(imlaey2uthmani)
        else:
            end = imlaey_wordspan.end

        # end is exclusive: the last index is `end -1`
        if end in imlaey2uthmani:
            if imlaey2uthmani[end - 1] == imlaey2uthmani[end]:
                raise PartOfUthmaniWord(
                    f"The Imlay Word is part of uthmani word, Sura: `{self.sura_idx + 1}`, Aya: `{self.aya_idx + 1}`, Imlaey Wordspan: ({start}, {end}), Uthmai Aya: {self.join_prefix.join(uthmani_words)}"
                )
        if (start > 0) and (imlaey2uthmani[start] == imlaey2uthmani[start - 1]):
            raise PartOfUthmaniWord(
                f"The Imlay Word is part of uthmani word, Sura: `{self.sura_idx + 1}`, Aya: `{self.aya_idx + 1}`, Imlaey Wordspan: ({start}, {end}), Uthmai Aya: {self.join_prefix.join(uthmani_words)}"
            )

        out_script = ""
        prev_uth_idx = -1
        for idx in range(start, end):
            if prev_uth_idx != imlaey2uthmani[idx]:
                out_script += uthmani_words[imlaey2uthmani[idx]]

                # Adding space Except for end idx
                if idx != end - 1:
                    out_script += self.join_prefix
            prev_uth_idx = imlaey2uthmani[idx]
        return out_script

    def _has_intersection(
        self, x: tuple[int, int] | None, y: tuple[int, int] | None
    ) -> bool:
        """
        Args:
            x: (tuple[int, int]): (start, end)
            y: (tuple[int, int]): (start, end)
        """
        if x is None or y is None:
            return False
        start = max(x[0], y[0])
        end = min(x[1], y[1])

        return end > start

    def imlaey_to_uthmani(
        self,
        imlaey_word_span: WordSpan,
        include_bismillah=False,
        include_istiaatha=False,
        include_sadaka=False,
        return_checks=False,
    ) -> str | Imlaey2uthmaniOutput:
        """return the uthmai script of the given imlaey script word indices

        Args:
            imlaey_word_span (WordSpan): the input imlay word ids in the Aya.
            Wordspan.start: the start word index, WordSpan.end: the end word index if is `None` means to the end of the aya

            include_bismillah (bool): If True, includes "Bismillah" (بسم الله الرحمن الرحيم)
                as part of the first ayah's encoding. Note: Bismillah is automatically
                included in Surah Al-Fatihah (as it is considered an ayah) and excluded
                in Surah At-Tawbah (no Bismillah in this surah).

            include_istiaatha (bool): If True, includes the Istiaatha (أعوذ بالله من الشيطان الرجيم)
                at the beginning of the surah (only for the first ayah).

            include_sadaka (bool): If True, appends "Sadaka Allahu Al-'Azeem" (صدق الله العظيم)
                after the last ayah of the surah.

        Returns:
            str: uthmani script if return_checks=False
            Imlaey2uthmaniOutput if return_checks=True

        Example:
            For Bismillah words ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَـٰنِ", "ٱلرَّحِيمِ"]
            WordSpan(start=1, end=3) → "ٱللَّهِ ٱلرَّحْمَـٰنِ"
        """
        encoding_out = self._encode_imlaey_to_uthmani(
            include_bismillah=include_bismillah,
            include_istiaatha=include_istiaatha,
            include_sadaka=include_sadaka,
        )

        uthmani_script = self._decode_uthmani(
            imlaey_wordspan=imlaey_word_span,
            imlaey2uthmani=encoding_out.imlaey2uthmani,
            uthmani_words=encoding_out.uthmani_words,
        )
        if return_checks:
            end_imlaey = (
                imlaey_word_span.end
                if imlaey_word_span.end is not None
                else len(encoding_out.imlaey_words)
            )
            input_iml_word_span = (imlaey_word_span.start, end_imlaey)
            has_quran = self._has_intersection(
                input_iml_word_span, encoding_out.aya_imlaey_span_words
            )
            if has_quran:
                quran_imlaey_word_start = max(
                    imlaey_word_span.start - encoding_out.aya_imlaey_span_words[0], 0
                )
                quran_imlaey_word_end = min(
                    end_imlaey - encoding_out.aya_imlaey_span_words[0],
                    encoding_out.aya_imlaey_span_words[1]
                    - encoding_out.aya_imlaey_span_words[0],
                )
                quran_start = QuranWordIndex(
                    imlaey=quran_imlaey_word_start,
                    uthmani=encoding_out.imlaey2uthmani[quran_imlaey_word_start],
                )
                quran_end = QuranWordIndex(
                    imlaey=quran_imlaey_word_end,
                    uthmani=encoding_out.imlaey2uthmani[quran_imlaey_word_end - 1] + 1,
                )
            else:
                quran_start = None
                quran_end = None

            return Imlaey2uthmaniOutput(
                imlaey=self.join_prefix.join(
                    encoding_out.imlaey_words[imlaey_word_span.start : end_imlaey]
                ),
                uthmani=uthmani_script,
                quran_start=quran_start,
                quran_end=quran_end,
                has_quran=has_quran,
                has_istiaatha=self._has_intersection(
                    input_iml_word_span,
                    encoding_out.istiaatha_imlaey_span_words,
                ),
                has_bismillah=self._has_intersection(
                    input_iml_word_span,
                    encoding_out.bismillah_imlaey_span_words,
                ),
                has_sadaka=self._has_intersection(
                    input_iml_word_span,
                    encoding_out.sadaka_imlaey_span_words,
                ),
            )
        return uthmani_script

    def get_by_imlaey_words(
        self,
        start: int,
        window: int,
        include_istiaatha=False,
        include_bismillah=False,
        include_sadaka=False,
    ) -> SegmentScripts:
        """returns the script format given start imlaey index (can be -ve) wiht length `window` words

        Args:
            start (int): the start index can be +ve or -ve. if -ve it will uses words
            from previous aya even in aya 1 sura 1 (circular looping)
            window (int): the number or imaley words to get

            include_bismillah (bool): If True, includes "Bismillah" (بسم الله الرحمن الرحيم)
                as part of the first ayah's encoding. Note: Bismillah is automatically
                included in Surah Al-Fatihah (as it is considered an ayah) and excluded
                in Surah At-Tawbah (no Bismillah in this surah).

            include_istiaatha (bool): If True, includes the Istiaatha (أعوذ بالله من الشيطان الرجيم)
                at the beginning of the surah (only for the first ayah).

            include_sadaka (bool): If True, appends "Sadaka Allahu Al-'Azeem" (صدق الله العظيم)
                after the last ayah of the surah.

        Returns:
            SegmentScripts container

        Example:
            start=-2, window=5 in Surah 1:1:
            - Retrieves last 2 words of previous ayah (114:6)
            - First 3 words of Surah 1:1
        """
        start_aya = self
        # making the start relative to the saved (self.start_imaley_words_idx)
        start += self.start_imlaey_word_idx
        if start < 0:
            new_start = start
            while new_start < 0:
                start_aya = start_aya.step(-1)
                encoding_out = start_aya._encode_imlaey_to_uthmani(
                    include_bismillah=include_bismillah,
                    include_istiaatha=include_istiaatha,
                    include_sadaka=include_sadaka,
                )
                new_start += len(encoding_out.imlaey_words)
            start = new_start

        # Moving the aya to the start position
        while True:
            encoding_out = start_aya._encode_imlaey_to_uthmani(
                include_bismillah=include_bismillah,
                include_istiaatha=include_istiaatha,
                include_sadaka=include_sadaka,
            )
            num_iml_words = len(encoding_out.imlaey_words)
            if start < num_iml_words:
                break
            start -= num_iml_words
            start_aya = start_aya.step(1)

        imlaey_str = ""
        uthmani_str = ""
        has_istiaatha = False
        has_bismillah = False
        has_quran = False
        has_sadaka = False
        first_time = True
        quran_word_start: QuranWordIndex | None = None
        quran_word_end: QuranWordIndex | None = None
        start_aya_idx = start_aya.get().aya_idx
        start_sura_idx = start_aya.get().sura_idx
        loop_aya = start_aya
        while window > 0:
            if imlaey_str != "":
                imlaey_str += self.join_prefix
            if uthmani_str != "":
                uthmani_str += self.join_prefix

            encoding_out = loop_aya._encode_imlaey_to_uthmani(
                include_bismillah=include_bismillah,
                include_istiaatha=include_istiaatha,
                include_sadaka=include_sadaka,
            )
            end = min(start + window, len(encoding_out.imlaey_words))
            iml2uth_out = loop_aya.imlaey_to_uthmani(
                WordSpan(start, end),
                include_istiaatha=include_istiaatha,
                include_bismillah=include_bismillah,
                include_sadaka=include_sadaka,
                return_checks=True,
            )

            if first_time and iml2uth_out.has_quran:
                first_time = False
                quran_word_start = iml2uth_out.quran_start

            imlaey_str += iml2uth_out.imlaey
            uthmani_str += iml2uth_out.uthmani
            has_quran = has_quran or iml2uth_out.has_quran
            has_istiaatha = has_istiaatha or iml2uth_out.has_istiaatha
            has_bismillah = has_bismillah or iml2uth_out.has_bismillah
            has_sadaka = has_sadaka or iml2uth_out.has_sadaka

            if has_quran:
                quran_word_end = iml2uth_out.quran_end
            end_aya_idx = loop_aya.get().aya_idx
            end_sura_idx = loop_aya.get().sura_idx

            # Steping
            assert end > start
            window -= end - start
            loop_aya = loop_aya.step(1)
            start = 0

        return SegmentScripts(
            imalaey=imlaey_str,
            uthmani=uthmani_str,
            start_span=(start_sura_idx, start_aya_idx, quran_word_start)
            if has_quran
            else None,
            end_span=(end_sura_idx, end_aya_idx, quran_word_end) if has_quran else None,
            has_quran=has_quran,
            has_istiaatha=has_istiaatha,
            has_bismillah=has_bismillah,
            has_sadaka=has_sadaka,
        )

    def step_by_imlaey_words(
        self,
        start: int,
        window: int,
        include_bismillah=False,
        include_istiaatha=False,
        include_sadaka=False,
    ) -> "Aya":
        """Navigates to new Aya position by word offset.

        Args:
            start (int): the start index can be +ve or -ve. if -ve it will uses words
                from previous aya even in aya 1 sura 1 (circular looping)
            window (int): the number or imaley words to get

            include_bismillah (bool): If True, includes "Bismillah" (بسم الله الرحمن الرحيم)
                as part of the first ayah's encoding. Note: Bismillah is automatically
                included in Surah Al-Fatihah (as it is considered an ayah) and excluded
                in Surah At-Tawbah (no Bismillah in this surah).

            include_istiaatha (bool): If True, includes the Istiaatha (أعوذ بالله من الشيطان الرجيم)
                at the beginning of the surah (only for the first ayah).

            include_sadaka (bool): If True, appends "Sadaka Allahu Al-'Azeem" (صدق الله العظيم)
                after the last ayah of the surah.

        Returns:
             New Aya object at calculated position

        Example:
            In Surah 1:1 with 4 words:
            step_by_imlaey_words(start=3, window=2) →
                Position at word 1 of next ayah (1:2)

        """
        step = self.start_imlaey_word_idx + start + window

        loop_aya = self
        # -ve step
        if step < 0:
            while step < 0:
                loop_aya = loop_aya.step(-1)
                encoding_out = loop_aya._encode_imlaey_to_uthmani(
                    include_bismillah=include_bismillah,
                    include_istiaatha=include_istiaatha,
                    include_sadaka=include_sadaka,
                )
                step += len(encoding_out.imlaey_words)
        # -ve step
        else:
            while True:
                encoding_out = loop_aya._encode_imlaey_to_uthmani(
                    include_bismillah=include_bismillah,
                    include_istiaatha=include_istiaatha,
                    include_sadaka=include_sadaka,
                )
                if step < len(encoding_out.imlaey_words):
                    break
                step -= len(encoding_out.imlaey_words)
                loop_aya = loop_aya.step(1)

        return Aya(
            sura_idx=loop_aya.sura_idx + 1,
            aya_idx=loop_aya.aya_idx + 1,
            start_imlaey_word_idx=step,
            quran_dict=self.quran_dict,
        )


@dataclass
class SearchItem:
    start_aya: Aya | None
    num_ayat: int
    imlaey_word_span: WordSpan | None
    uthmani_script: str
    has_bismillah: bool = False
    has_istiaatha: bool = False
    """
    start_aya (Aya): the start aya of the first search

    num_aya: (int): number of ayat that is included in the search item

    has_bismillah (bool): True if the search item has bismliilah
        (not the Aya in El-Fatiha of in the Alnaml)

    has_istiaatha (bool): True if the search item has istiaatha

    imlaey_word_span (WordSpan):
        start: the start word idx of the imlaey scriptin thestart_aya
        end: the end imlaey_idx of the imlaey (start_aya + num_ayat - 1)

    uthmani_script (str) the equvilent uthmani script of the given imlaey script
    if istiaatha is only will return:
        start_aya=None, num_ayat=None, imlaey_word_span=None, has_bismillah=None
    """

    def __str__(self):
        out_str = ""
        if self.start_aya:
            out_str += f"start_aya(sura_idx={self.start_aya.get().sura_idx}, aya_idx={self.start_aya.get().aya_idx})"
        else:
            out_str += f"start_aya(sura_idx={None}, aya_idx={None})"
        out_str += f", num_ayat={self.num_ayat}"
        out_str += f", uthmnai_script={self.uthmani_script}"
        out_str += f", has_istiaatha={self.has_istiaatha}"
        out_str += f", has_bismillah={self.has_bismillah}"
        out_str += f", imlaey_word_span={self.imlaey_word_span}"

        return out_str


# TODO: Add Examples
def search(
    text: str,
    start_aya: Aya = Aya(1, 1),
    window: int = 2,
    suffix=" ",
    **kwargs,
) -> list[SearchItem]:
    """searches the Holy Quran of Imlaey script to match the given text

    searches the Holy quran for a given Imlaey text of specifc window arround
    the `start_aya` applying some filters of the search string

    Example:
        >> results = search('الحمد لله',
            start_aya=Aya(1, 1), window=4, ignore_tashkeel=True)
        >> results
    Args:
        text (str): the text to search with (expected with imlaey script)

        start_aya (Aya): The Pivot Aya to set Search with.

        winodw (int): the search winodw:
        [start_aya - winowd //2, start_aya + winodw //2]

        suffix (str): the suffix that sperate the quran words either imlaey or uthmani
        the rest of **kwargs are from normalize_aya function below
    Returns:
        list[SearchItem]: Every SearchItem is:
        * start_aya (Aya): the start aya of the first search

        * num_aya: (int): number of ayat that is included in the search item

        * has_bismillah (bool): True if the search item has bismliilah
        (not the Aya in El-Fatiha of in the Alnaml)

        * has_istiaatha (bool): True if the search item has istiaatha

        * imlaey_word_span (WordSpan):
        start: the start word idx of the imlaey scriptin thestart_aya
        end: the end imlaey_idx of the imlaey (start_aya + num_ayat - 1)

        * uthmani_script (str) the equvilent uthmani script of the given imlaey script
        NOTE: if istiaatha is only will return:
        start_aya=None, num_ayat=None, imlaey_word_span=None, has_bismillah=None
    """
    normalized_text: str = normalize_aya(text, remove_spaces=True, **kwargs)
    if normalized_text == "":
        return []

    # Prepare ayat within [-window/2, window/2]
    loop_aya = start_aya.step(-window // 2)

    # ----------------------------------
    # Checking for Itiaatha
    # ----------------------------------
    # NOTE: Assuming Istiaatha is at the first only
    has_istiaatha = False
    istiaatha_imlaey_words = normalize_aya(
        start_aya.get().istiaatha_imlaey,
        remove_spaces=False,
        **kwargs,
    ).split(suffix)
    istiaatha_imlaey_str = "".join(istiaatha_imlaey_words)
    re_span = re.search(istiaatha_imlaey_str, normalized_text)
    if re_span:
        normalized_text = normalized_text[re_span.span()[1] :]
        has_istiaatha = True
        if normalized_text == "":
            # return istiaatha only
            return [
                SearchItem(
                    start_aya=None,
                    num_ayat=0,
                    imlaey_word_span=None,
                    has_bismillah=False,
                    has_istiaatha=has_istiaatha,
                    uthmani_script=start_aya.get().istiaatha_uthmani,
                )
            ]

    found = []
    for bismillah_flag in [False, True]:
        aya_imlaey_words, aya_imlaey_str = _get_imlaey_words_and_str(
            start_aya=loop_aya,
            window=window,
            suffix=suffix,
            include_bismillah=bismillah_flag,
            **kwargs,
        )

        for re_search in re.finditer(normalized_text, aya_imlaey_str):
            if re_search is not None:
                span = _get_words_span(
                    start=re_search.span()[0],
                    end=re_search.span()[1],
                    words_list=aya_imlaey_words,
                )
                if span is not None:
                    start_vertex, end_vertex = span
                    found.append(
                        SearchItem(
                            start_aya=loop_aya.step(start_vertex.aya_idx),
                            num_ayat=end_vertex.aya_idx - start_vertex.aya_idx + 1,
                            imlaey_word_span=WordSpan(
                                start=start_vertex.word_idx, end=end_vertex.word_idx
                            ),
                            has_bismillah=bismillah_flag,
                            has_istiaatha=has_istiaatha,
                            uthmani_script="",
                        )
                    )
                    found[-1].uthmani_script = _get_uthmani_of_result_item(
                        found[-1], suffix=suffix
                    )
        if found != []:
            # add istiaatah uthamni script
            if has_istiaatha:
                for item in found:
                    item.uthmani_script = (
                        start_aya.get().istiaatha_uthmani + suffix + item.uthmani_script
                    )
            return found

    return found


def normalize_aya(
    text: str,
    remove_spaces=True,
    ignore_hamazat=False,
    ignore_alef_maksoora=True,
    ignore_taa_marboota=False,
    normalize_taat=False,
    remove_small_alef=True,
    remove_tashkeel=False,
) -> str:
    """Apply filters to match input Kwargs on **Imlaey** text

    Args:
        remove_spaces (bool): remove spaces for text

        ignore_hamazat (bool): making all hamazat equal (أ, آ, إ ء, ئ, ؤ) = (ء)
        alphabet.imlaey.hmazat -> alphabet.imlaey.

        ignore_alef_maksoora (bool): (ى) -> (ا).
        alphabet.imlaey.alef_maksoora = alphabet.imlaey.alef

        ignore_taa_marboota (bool): (ة) -> (ه).
        alphabet.imlaey.taa_motaterfa -> alphabet.imlaey.haa

        normalize_taat (bool): (ة) -> (ت).
        alphabet.imlaey.taa_marboota = alphabet.taa_mabsoota

        NOTE: We can not use `ignore_taaa_marboota` and `normalize_taaat` at
        the same time

        remove_small_alef (bool): remove small alef "ٰ" in
        alphabet.imlaey.small_alef (alef khingarai)

        remove_tashkeel (bool): remove tashkeel: "ًٌٍَُِّْ" in alphabet.imlaey.tashkeel

        Return:
            str: the normalied imlaey text
    """
    assert not (ignore_taa_marboota and normalize_taat), (
        "You can not `ignore_taa_marboota` and `normaize_taat` at the same time"
    )

    norm_text = text

    # TODO: Ingonre alef as hamza

    if remove_spaces:
        norm_text = re.sub(r"\s+", "", norm_text)

    if ignore_alef_maksoora:
        norm_text = re.sub(alpha.imlaey.alef_maksoora, alpha.imlaey.alef, norm_text)

    if ignore_hamazat:
        norm_text = re.sub(f"[{alpha.imlaey.hamazat}]", alpha.imlaey.hamza, norm_text)

    if ignore_taa_marboota:
        norm_text = re.sub(
            f"[{alpha.imlaey.taa_marboota}]",
            alpha.imlaey.haa,
            norm_text,
        )

    if normalize_taat:
        norm_text = re.sub(
            f"[{alpha.imlaey.taa_marboota}]",
            alpha.imlaey.taa_mabsoota,
            norm_text,
        )

    if remove_small_alef:
        norm_text = re.sub(alpha.imlaey.small_alef, "", norm_text)

    if remove_tashkeel:
        norm_text = re.sub(f"[{alpha.imlaey.tashkeel}]", "", norm_text)

    return norm_text


def _get_words_span(
    start: int, end: int, words_list: list[list[str]]
) -> tuple[Vertex, Vertex] | None:
    """
    return the word indices at every word boundary only not inside the word:
    which means:
    * start character is at the beginning of the word
    * end character is at the end of the word + 1
    EX: start = 0, end = 8, words_list=[['aaa', 'bbb',], ['cc', 'ddd']]
                                          ^                 ^
                                          0               8 - 1
    return (start, end)
    (start.aya_idx=0, start.word_idx=0, end. aya_idx=1, end.word_idx=0 + 1)

    return None if:
        * start not at the beginning of the word.
        * end is not at (end + 1) of the word.
        * start >= end

    Args:
        start (int): the start char idx
        end (int): the end char idx + 1
        words_list (list[list[str]]): given words

    return: WordSpan:
        start: the start idx of the word in "words"
        end: (end_idx + 1) of the word in "words"
        if valid boundary else None
    """

    def _get_start_span(start_char: int) -> tuple[int, int] | None:
        chars_count = 0
        for aya_idx in range(len(words_list)):
            for word_idx in range(len(words_list[aya_idx])):
                if start_char == chars_count:
                    return aya_idx, word_idx
                chars_count += len(words_list[aya_idx][word_idx])
            aya_idx += 1
        return None

    def _get_end_span(
        end_char: int, chars_count=0, start_aya_idx=0, start_word_idx=0
    ) -> tuple[int, int] | None:
        for aya_idx in range(start_aya_idx, len(words_list)):
            for word_idx in range(start_word_idx, len(words_list[aya_idx])):
                chars_count += len(words_list[aya_idx][word_idx])
                if end_char == chars_count:
                    return aya_idx, word_idx + 1
            start_word_idx = 0
        return None

    # print('start=', start, ', end=', end)
    span = _get_start_span(start)
    # print(f'start=({span})')
    if span is None:
        return None
    start_aya_idx, start_word_idx = span

    span = _get_end_span(
        end,
        chars_count=start,
        start_aya_idx=start_aya_idx,
        start_word_idx=start_word_idx,
    )
    # print(f'end=({span})')
    if span is None:
        return None
    end_aya_idx, end_word_idx = span
    return (
        Vertex(aya_idx=start_aya_idx, word_idx=start_word_idx),
        Vertex(aya_idx=end_aya_idx, word_idx=end_word_idx),
    )


def _get_uthmani_of_result_item(search_item: SearchItem, suffix=" ") -> str:
    """
    add uthmani script of the imlaey script found in the SearchItem
    """
    # parsing spans
    wordspans = [WordSpan(0, None) for _ in range(search_item.num_ayat)]
    wordspans[0].start = search_item.imlaey_word_span.start
    wordspans[-1].end = search_item.imlaey_word_span.end

    uthmani_str = ""
    for idx, aya in enumerate(
        search_item.start_aya.get_ayat_after(num_ayat=search_item.num_ayat)
    ):
        uthmani_str += aya.imlaey_to_uthmani(
            wordspans[idx],
            include_bismillah=search_item.has_bismillah,
        )
        uthmani_str += suffix
    # removing last suffix from the nd
    uthmani_str = uthmani_str[: -len(suffix)]

    return uthmani_str


def _get_imlaey_words_and_str(
    start_aya: Aya,
    window: int,
    include_bismillah=False,
    suffix=" ",
    **kwargs,
) -> tuple[list[list[str]], str]:
    """
    return (words, scipt): The imlaey script either of multiple ayat
        words: 2D list dimention(0) is of length of number of ayat, dimention(1)
            is the aya words
        script: the joined script of ayat without spaces
    """
    aya_imlaey_words: list[list[str]] = []
    aya_imlaey_str = ""
    for aya in start_aya.get_ayat_after(num_ayat=window + 1):
        aya_words = []

        # Including Bismillah at The start of sura except for:
        # Alfatiha [is an Aya] and Al tuoba
        if include_bismillah and (aya.get().bismillah_imlaey is not None):
            aya_words += normalize_aya(
                aya.get().bismillah_imlaey,
                remove_spaces=False,
                **kwargs,
            ).split(suffix)

        # Aya Words
        aya_words += normalize_aya(
            aya.get().imlaey,
            remove_spaces=False,
            **kwargs,
        ).split(suffix)
        aya_imlaey_words.append(aya_words)

        # imlaey String With spaces removed
        aya_imlaey_str += re.sub(r"\s+", "", "".join(aya_words))
    return aya_imlaey_words, aya_imlaey_str
