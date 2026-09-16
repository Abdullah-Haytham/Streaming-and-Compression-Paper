from dataclasses import dataclass
import math
import logging
from typing import Optional

import Levenshtein as lv

from .utils import normalize_aya, Aya, SegmentScripts, QuranWordIndex, PartOfUthmaniWord


def estimate_window_len(text: str, winodw_words: int) -> tuple[int, int]:
    return (max(1, int(len(text) / 9)), math.ceil(len(text) / 2))


def estimate_overlap(text: str, prev_text: str | None, max_overlap: int) -> int:
    if prev_text is None:
        return 0

    # suppose that the word has 2 characters
    return min(int(len(prev_text) / 3.5), max_overlap)


@dataclass
class BestSegment:
    start: int
    window: int
    segment_scripts: SegmentScripts | None = None
    ratio: float = 0.0
    bisimillah: bool = False


def get_match_ratio(ref_text: str, other_text: str) -> float:
    # ratio = lv.ratio(ref_text, other_text)
    return 1 - (min(lv.distance(ref_text, other_text), len(ref_text)) / len(ref_text))


def tasmeea_sura(
    text_segments: list[str],
    sura_idx: int,
    pivot_list: Optional[list[str] | None] = None,
    overlap_words: int = 6,
    window_words=30,
    acceptance_ratio: float = 0.5,
    include_istiaatha=True,
    include_bismillah=True,
    include_sadaka=True,
    **kwargs,
) -> list[tuple[SegmentScripts | None, float]]:
    """Returns the best matching quracic script for every text part

    Args:
        pivot_list (list[str] | None): A list where each element is either "pivot" or an empty string "".
            "pivot" indicates that the corresponding item is the first segment when the text is split into multiple parts.
            An empty string ("") means the item is not the first segment.
            This helps identify which segments are the starting points in a split text.


        Note:
            - We only support Istiaatha to be spearate segment not connected to other ayat or bismillah
            - We only support Sadaka to be spearate segment not connected to other ayat or bismillah
    """

    def _check_segment(
        _best: BestSegment,
        _aya,
        _norm_text: str,
        _start,
        _window,
        _istiaatha=False,
        _bismillah=False,
        _sadaka=False,
    ) -> BestSegment | None:
        try:
            segment_scripts = _aya.get_by_imlaey_words(
                start=_start,
                window=_window,
                include_istiaatha=_istiaatha,
                include_bismillah=_bismillah,
                include_sadaka=_sadaka,
            )
            aya_imalaey_str = normalize_aya(segment_scripts.imalaey, **kwargs)
            match_ratio = get_match_ratio(_norm_text, aya_imalaey_str)
            if (match_ratio > _best.ratio) or (
                match_ratio == _best.ratio and abs(_start) < abs(_best.start)
            ):
                _best.segment_scripts = segment_scripts
                _best.ratio = match_ratio
                _best.bisimillah = _bismillah
                if _istiaatha or _sadaka:
                    _best.window = 0
                    _best.start = 0
                else:
                    _best.window = _window
                    _best.start = _start
                return _best
            else:
                return None
        except PartOfUthmaniWord:
            return None

    assert overlap_words >= 0
    if pivot_list is None:
        pivot_list = ["pivot"] * len(text_segments)

    kwargs["remove_spaces"] = True
    kwargs["remove_tashkeel"] = True
    aya = Aya(
        sura_idx=sura_idx,
    )
    last_aya = aya.step(aya.get().num_ayat_in_sura - 1)
    outputs = []
    prev_norm_text = None
    penalty = 0
    for idx, text_seg in enumerate(text_segments):
        norm_text = normalize_aya(text_seg, **kwargs)
        min_winodw_len, max_windwo_len = estimate_window_len(norm_text, window_words)
        # overlap_len = estimate_overlap(norm_text, prev_norm_text, overlap_words)
        overlap_len = overlap_words
        # the case that the overloap is too big (16) and window is too small (6) end_words = -10
        # so we can not check the true aya part instead end_words = 16 + 6 = 22
        start_words = -(overlap_len + penalty)
        end_words = (
            overlap_len + max(window_words - overlap_len, max_windwo_len) + penalty
        )

        best = BestSegment(
            segment_scripts=None,
            ratio=0.0,
            start=-overlap_len,
            window=min_winodw_len,
            bisimillah=False,
        )
        logging.debug(
            f"{idx} -> Start Span{aya.get().sura_idx, aya.get().aya_idx, aya.get_start_imlaey_word_idx()}, Text: {text_seg}, Start: {start_words}, End: {end_words}, Min Window: {min_winodw_len}, Max Window: {max_windwo_len}, Overlap: {overlap_len}, Penlety: {penalty}"
        )
        if len(norm_text) > 0:
            # istiaatha at the first
            if idx == 0 and include_istiaatha:
                out = _check_segment(
                    _best=best,
                    _aya=aya,
                    _norm_text=norm_text,
                    _start=0,
                    _window=5,
                    _istiaatha=True,
                    _bismillah=False,
                    _sadaka=False,
                )
                if out:
                    best = out
            # sadaka only at the last aya
            elif (idx + 1) == len(text_segments) and include_sadaka:
                sadaka_start = (
                    len(last_aya.get().imlaey_words)
                    - last_aya.get_start_imlaey_word_idx()
                )
                out = _check_segment(
                    _best=best,
                    _aya=last_aya,
                    _norm_text=norm_text,
                    _start=sadaka_start,
                    _window=3,
                    _istiaatha=False,
                    _bismillah=False,
                    _sadaka=True,
                )
                if out:
                    best = out

            # Initializing step words with min_window_len if not acceptable match
            for loop_start in range(start_words, end_words):
                # looping over all available windows
                bismillah = (
                    aya.get().sura_idx not in {1, 9}
                    and include_bismillah
                    and aya.get().aya_idx == 1
                ) or (idx == 1 and outputs[0][0] is None)

                for loop_window_len in range(min_winodw_len, max_windwo_len + 1):
                    out = _check_segment(
                        _best=best,
                        _aya=aya,
                        _norm_text=norm_text,
                        _start=loop_start,
                        _window=loop_window_len,
                        _istiaatha=False,
                        _bismillah=bismillah,
                        _sadaka=False,
                    )
                    if out:
                        best = out

            # reset penalities for the next loop
            penalty = 0

        if best.segment_scripts is None:
            penalty = max_windwo_len
            outputs.append((None, best.ratio))
            aya = aya.step_by_imlaey_words(
                start=-overlap_len,
                window=int((min_winodw_len + max_windwo_len) / 2),
                include_bismillah=False,
            )
        elif best.ratio < acceptance_ratio:
            penalty = max_windwo_len
            outputs.append((None, best.ratio))
            aya = aya.step_by_imlaey_words(
                start=-overlap_len,
                window=int((min_winodw_len + max_windwo_len) / 2),
                include_bismillah=False,
            )
        else:
            outputs.append((best.segment_scripts, best.ratio))
            aya = aya.step_by_imlaey_words(
                start=best.start,
                window=best.window,
                include_bismillah=best.bisimillah,
            )

        # some text segments are spaned accorss multiple items
        # the pivot is the fistt item may be the logest one
        if pivot_list[idx] == "pivot":
            prev_norm_text = norm_text

    return outputs


def compute_prefix_function(pattern):
    pi = [0] * len(pattern)
    k = 0
    for q in range(1, len(pattern)):
        while k > 0 and pattern[k] != pattern[q]:
            k = pi[k - 1]
        if pattern[k] == pattern[q]:
            k += 1
        pi[q] = k
    return pi


def merge_text(texts: list[str]) -> str:
    if not texts:
        return ""
    merged = texts[0]
    for i in range(1, len(texts)):
        next_text = texts[i]
        L = min(len(merged), len(next_text))
        s_tail = merged[-L:]
        pi = compute_prefix_function(next_text)
        state = 0  # number of matched chars
        for char in s_tail:
            # going batch to tha last matching prefix
            while state > 0 and next_text[state] != char:
                state = pi[state - 1]
            if next_text[state] == char:
                state += 1
        # if state < min_merge_chars:
        #     raise SmallTarteelOverlap("Very Small overlap to merge on")
        merged += next_text[state:]
    return merged


def merge_segment_scritps(
    seg_scrips: list[SegmentScripts | None],
) -> SegmentScripts | None:
    # TODO:
    # Potensional bug if the span are not in order
    # Istiaatha, Bsimillah, and Saadka support
    if any(s is None for s in seg_scrips):
        return None

    assert any(s.has_quran for s in seg_scrips), (
        "We support only merging scripts with quranic verses"
    )
    assert not (
        any(s.has_istiaatha for s in seg_scrips)
        or any(s.has_bismillah for s in seg_scrips)
        or any(s.has_sadaka for s in seg_scrips)
    ), "We only support quranic verses"

    start_span = seg_scrips[0].start_span
    end_span = seg_scrips[-1].end_span

    assert start_span is not None and end_span is not None
    start_aya = Aya(
        start_span[0], start_span[1], start_imlaey_word_idx=start_span[2].imlaey
    )

    # Computing Window
    loop_aya = start_aya
    window = 0
    while True:
        if (
            loop_aya.get().sura_idx == end_span[0]
            and loop_aya.get().aya_idx == end_span[1]
        ):
            window += end_span[2].imlaey - loop_aya.get_start_imlaey_word_idx()
            break
        else:
            window += (
                len(loop_aya._encode_imlaey_to_uthmani().imlaey_words)
                - loop_aya.get_start_imlaey_word_idx()
            )
            loop_aya = loop_aya.step(1)

    return start_aya.get_by_imlaey_words(start=0, window=window)


def tasmeea_sura_multi_part(
    text_segments: list[list[str]],
    sura_idx: int,
    overlap_words: int = 6,
    window_words=30,
    acceptance_ratio: float = 0.5,
    include_istiaatha=True,
    include_bismillah=True,
    include_sadaka=True,
    multi_part_truncation_words=2,
    **kwargs,
) -> list[tuple[SegmentScripts | None, float]]:
    """Returns the best matching quracic script for every text part

    Note:
        - We only support Istiaatha to be spearate segment not connected to other ayat or bismillah
        - We only support Sadaka to be spearate segment not connected to other ayat or bismillah
    """

    def _trim_multi_segs(_segs: list[str], _truncation_words: int = 2) -> list[str]:
        # Removing trailing words and starting words
        # to avoide (part of word wrong transcrips)
        # Not removing the start of the first transcript
        words = _segs[0].split(" ")
        _segs[0] = " ".join(words[: len(words) - _truncation_words])
        for idx in range(1, len(_segs) - 1):
            words = _segs[idx].split(" ")
            _segs[idx] = " ".join(
                words[_truncation_words : len(words) - _truncation_words]
            )
        # Not removing the tail of the last transcript
        _segs[-1] = " ".join(_segs[-1].split(" ")[_truncation_words:])
        return _segs

    to_process_segs = []
    pivot_list: list[str] = []
    # (start_index, num_segments)
    multi_part_ids: dict[int, int] = {}
    for seg in text_segments:
        if len(seg) > 1:
            pivot_list.append("pivot")
            pivot_list += [""] * (len(seg) - 1)
            multi_part_ids[len(to_process_segs)] = len(seg)
            seg = _trim_multi_segs(seg, _truncation_words=multi_part_truncation_words)
        else:
            pivot_list.append("pivot")

        to_process_segs += seg

    outs_parted = tasmeea_sura(
        text_segments=to_process_segs,
        sura_idx=sura_idx,
        pivot_list=pivot_list,
        include_istiaatha=include_istiaatha,
        include_bismillah=include_bismillah,
        include_sadaka=include_sadaka,
        window_words=window_words,
        overlap_words=overlap_words,
        acceptance_ratio=acceptance_ratio,
        **kwargs,
    )

    outs = []
    idx = 0
    while idx < len(outs_parted):
        if idx in multi_part_ids:
            multi_seg = outs_parted[idx : idx + multi_part_ids[idx]]
            seg = merge_segment_scritps([o[0] for o in multi_seg])
            avg_score = sum([o[1] for o in multi_seg]) / len(multi_seg)
            outs.append((seg, avg_score))
            idx += len(multi_seg)
        else:
            outs.append(outs_parted[idx])
            idx += 1

    return outs


def check_sura_missing_parts(
    sura_idx: int,
    fixed_segments: list[SegmentScripts],
) -> list[SegmentScripts]:
    """we are checkint quranic verses only (not includes istiaatha, bismillah, or sadaka)"""

    def _find_missings(
        _start_aya: Aya,
        _start: tuple[int, int, QuranWordIndex],
        _end: tuple[int, int, QuranWordIndex],
    ) -> list[SegmentScripts]:
        if _end[0] < _start[0]:
            return []
        elif _end[0] >= _start[0] and _end[1] < _start[1]:
            return []
        elif (
            _start[0] == _end[0]
            and _start[1] == _end[1]
            and _end[2].imlaey < _start[2].imlaey
        ):
            return []

        _missings = []
        _start_aya = _start_aya.set_new(
            sura_idx=_start[0],
            aya_idx=_start[1],
            start_imlaey_word_idx=_start[2].imlaey,
        )
        _loop_aya = _start_aya
        while True:
            if (_loop_aya.get().sura_idx == _end[0]) and (
                _loop_aya.get().aya_idx == _end[1]
            ):
                window = _end[2].imlaey - _loop_aya.get_start_imlaey_word_idx()
                if window > 0:
                    miss_seg = _loop_aya.get_by_imlaey_words(
                        start=0,
                        window=window,
                    )
                    _missings.append(miss_seg)
                break

            # else
            window = (
                len(_loop_aya._encode_imlaey_to_uthmani().imlaey_words)
                - _loop_aya.get_start_imlaey_word_idx()
            )
            if window > 0:
                miss_seg = _loop_aya.get_by_imlaey_words(start=0, window=window)
                _missings.append(miss_seg)
            _loop_aya = _loop_aya.step(1)

        return _missings

    start_aya = Aya(sura_idx=sura_idx)
    start = (
        start_aya.get().sura_idx,
        start_aya.get().aya_idx,
        QuranWordIndex(imlaey=0, uthmani=0),
    )
    missings: list[SegmentScripts] = []
    actual_segments = [s for s in fixed_segments if s is not None]
    last_quran_seg: SegmentScripts | None = None
    for seg in actual_segments:
        if seg.has_quran:
            missings += _find_missings(
                _start_aya=start_aya,
                _start=start,
                _end=seg.start_span,
            )
            start = seg.end_span
            last_quran_seg = seg

    assert last_quran_seg is not None, "No Quan segments"
    if last_quran_seg.end_span[0] != sura_idx:
        end = last_quran_seg.end_span
    else:
        last_aya = start_aya.step(start_aya.get().num_ayat_in_sura - 1)
        last_aya_encoding = last_aya._encode_imlaey_to_uthmani()
        iml_words = len(last_aya_encoding.imlaey_words)
        end = (
            start_aya.get().sura_idx,
            start_aya.get().num_ayat_in_sura,
            QuranWordIndex(
                imlaey=iml_words,
                uthmani=last_aya_encoding.imlaey2uthmani[iml_words - 1] + 1,
            ),
        )

    missings += _find_missings(_start_aya=start_aya, _start=start, _end=end)

    return missings
