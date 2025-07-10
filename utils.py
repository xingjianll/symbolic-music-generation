from miditok import TokenizerConfig, PerTok

CONTEXT_SIZE = 4096

def get_tokenizer(load=True):
    config = TokenizerConfig(
        num_velocities=8,
        use_velocities=True,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
        use_pitch_intervals=False,
        use_programs=False,
        use_pitchdrum_tokens=False,
        ticks_per_quarter=320,
        use_microtiming=False,
        max_microtiming_shift=0.125
    )
    tokenizer = PerTok(config)
    if load:
        tokenizer.from_pretrained("xingjianll/midi-tokenizer-v2")

    return tokenizer