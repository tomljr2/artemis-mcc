from collections import Counter

from data.extract_benchmark import BENCHMARK


def test_twenty_different_reports():
    ids = [ntrs_id for ntrs_id, _, _ in BENCHMARK]
    assert len(ids) == 20 and len(set(ids)) == 20


def test_every_decade_from_the_1960s_on_is_represented():
    decades = Counter(year // 10 * 10 for _, year, _ in BENCHMARK)
    assert sorted(decades) == [1960, 1970, 1980, 1990, 2000, 2010, 2020]
    assert all(count >= 2 for count in decades.values())
