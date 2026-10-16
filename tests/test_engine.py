import pytest

from offline_translate.engine import add_period_limit, split_sentences, strip_period, with_period


@pytest.mark.parametrize("text, expected", [
    ("Hello.", [["Hello."]]),
    ("Hi. How are you? Fine!", [["Hi.", "How are you?", "Fine!"]]),
    ("No split.here 3.5 times", [["No split.here 3.5 times"]]),
    ("one.\ntwo.", [["one."], ["two."]]),
    ("a\n\nb", [["a"], [], ["b"]]),
    ("שלום. מה שלומך?", [["שלום.", "מה שלומך?"]]),
    ("你好。 再见", [["你好。", "再见"]]),
    ("  padded.  ", [["padded."]]),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_add_period_limit_from_real_spec_format():
    spec = (b'\n\x02en\x12\x02iw"ePreprocAddPeriod/add_period_4:max_sent_len=4:skip_in_training=true'
            b':use_wpm_tokens=false:mode=annotate"\x1cPreprocB2A/b2a:mode=tokenize')
    assert add_period_limit(spec) == 4
    assert add_period_limit(b'"\x1cPreprocB2A/b2a:mode=tokenize') is None


@pytest.mark.parametrize("sentence, limit, expected", [
    ("car", 4, ("car.", True)),
    ("one two three four", 4, ("one two three four.", True)),
    ("one two three four five", 4, ("one two three four five", False)),
    ("car.", 4, ("car.", False)),
    ("really?", 4, ("really?", False)),
    ("car", None, ("car", False)),
])
def test_with_period(sentence, limit, expected):
    assert with_period(sentence, limit) == expected


@pytest.mark.parametrize("text, expected", [("מכונית.", "מכונית"), ("車。", "車"), ("hi!", "hi!"), ("", "")])
def test_strip_period(text, expected):
    assert strip_period(text) == expected
