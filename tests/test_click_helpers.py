import gzip

import pytest

from hector_ml.click_helpers import smart_open


@pytest.mark.parametrize(
    "filename,content",
    [
        ("foo.txt", b"foo\n"),
        ("foo.txt.gz", gzip.compress(b"foo\n")),
    ],
)
def test_smart_open(tmp_path, filename, content):
    (tmp_path / filename).write_bytes(content)
    with smart_open(tmp_path / filename, "rt") as f:
        assert f.read() == "foo\n"


def test_smart_open_stdio(capsysbinary):
    with smart_open("-", "wt") as f:
        f.write("foo\n")
        f.flush()
    assert capsysbinary.readouterr() == (b"foo\n", b"")
