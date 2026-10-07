from pretrain.plot_losses import parse_log

CURRENT = """gpt: 811,904 parameters
step     0 | train 4.655 | val transcript 4.681 | val book 4.641 |     3s
           saved new best to checkpoints\\gpt_nasa.pt
step   500 | train 1.546 | val transcript 1.640 | val book 1.701 |    21s
--- generated ---
step 1000 is not a progress line, just generated text
"""

OLDER = """step     0 | train loss 4.485 | val loss 4.483 |     2s
step   500 | train loss 1.235 | val loss 1.452 |    19s
"""


def test_reads_every_series_from_the_current_printout():
    assert parse_log(CURRENT) == {
        "train": [(0, 4.655), (500, 1.546)],
        "val transcript": [(0, 4.681), (500, 1.640)],
        "val book": [(0, 4.641), (500, 1.701)],
    }


def test_reads_the_older_printout_format_too():
    assert parse_log(OLDER) == {
        "train": [(0, 4.485), (500, 1.235)],
        "val": [(0, 4.483), (500, 1.452)],
    }
