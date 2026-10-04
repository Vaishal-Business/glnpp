from galnp_s.capture import channels as C


def test_lengths():
    assert len(C.CHANNEL_NAMES) == 52
    assert len(C.UPPER_GROUP) == 21
    assert len(C.LOWER_GROUP) == 32
    assert len(C.CRITICAL_CHANNELS) == 15


def test_groups_are_subsets():
    S = set(C.CHANNEL_NAMES)
    assert set(C.UPPER_GROUP) <= S
    assert set(C.LOWER_GROUP) <= S
    assert set(C.CRITICAL_CHANNELS) <= S


def test_cheek_squint_in_both_groups():
    assert "cheekSquintLeft" in C.UPPER_GROUP and "cheekSquintLeft" in C.LOWER_GROUP
    assert "cheekSquintRight" in C.UPPER_GROUP and "cheekSquintRight" in C.LOWER_GROUP


def test_no_duplicates_in_groups():
    assert len(set(C.UPPER_GROUP)) == len(C.UPPER_GROUP)
    assert len(set(C.LOWER_GROUP)) == len(C.LOWER_GROUP)