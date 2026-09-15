"""Audio decoding, label encoding, dataset streaming, and augmentation."""


def decode_audio(audio_field, target_sr=16000):
    import io
    import soundfile as sf
    buf = io.BytesIO(audio_field["bytes"])
    waveform, sr = sf.read(buf, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, sr


def encode_labels(sample, vocab, levels):
    encoded = {}
    phoneme_str = sample.get("phonemes", "") or ""
    phoneme_vocab = vocab.get("phonemes", {})
    encoded["phonemes"] = [phoneme_vocab[c] for c in phoneme_str if c in phoneme_vocab]

    sifat = sample.get("sifat", []) or []
    for level in levels:
        if level == "phonemes":
            continue
        level_vocab = vocab.get(level, {})
        ids = []
        for entry in sifat:
            value = entry.get(level, "")
            if value and value in level_vocab:
                ids.append(level_vocab[value])
        encoded[level] = ids
    return encoded


def stream_samples(moshaf, num_samples):
    """Lazy interleaved stream over one or more moshaf subsets.

    Multiple moshafs are interleaved with equal probability and the stream
    stops once every constituent dataset has been exhausted (or ``num_samples``
    is reached, whichever comes first).
    """
    from datasets import Audio, load_dataset, interleave_datasets

    if isinstance(moshaf, str):
        moshaf = [moshaf]

    print(f"  streaming {num_samples} samples across {len(moshaf)} moshaf(s): {moshaf}")

    streams = []
    for name in moshaf:
        ds = load_dataset(
            "obadx/muaalem-annotated-v3", name,
            split="train", streaming=True,
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        streams.append(ds)

    if len(streams) == 1:
        combined = streams[0]
    else:
        combined = interleave_datasets(
            streams,
            probabilities=[1 / len(streams)] * len(streams),
            seed=42,
            stopping_strategy="all_exhausted",
        )

    return combined.take(num_samples)


def build_augmentation():
    try:
        from audiomentations import Compose, TimeStretch, GainTransition, AddGaussianNoise
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
            TimeStretch(min_rate=0.8, max_rate=1.5, p=0.4),
            GainTransition(min_gain_db=-6, max_gain_db=6, p=0.4),
        ])
    except ImportError:
        print("  audiomentations not installed -- skipping augmentation")
        return None
