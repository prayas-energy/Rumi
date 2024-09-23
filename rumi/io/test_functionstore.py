from rumi.io import functionstore as fs
import pandas as pd
import pytest

def test_isnone():
    assert not fs.isnone(pd.DataFrame())
    assert fs.isnone(None)


def test_is_empty_or_none():
    assert fs.is_empty_or_none(None)
    assert fs.is_empty_or_none(pd.DataFrame())

    
def test_override_dataframe():
    df1 = pd.DataFrame({"a":list("ABCDEF"),
                        "b":range(6)})
    df2 = pd.DataFrame({"a":list("ACF"),
                        "b":[5,5,5]})

    df3 = fs.override_dataframe(df1, df2, ['a'])
    df4 = pd.DataFrame({"a":list("ABCDEF"),
                        "b":[5, 1, 5, 3, 4, 5]})

    assert set(df3.itertuples(index=False, name=None)) == set(df4.itertuples(index=False, name=None))

    df5 = fs.override_dataframe(df1, pd.DataFrame(), "a")
    assert set(df5.itertuples(index=False, name=None)) == set(df1.itertuples(index=False, name=None))

    df5 = fs.override_dataframe(df1, None, "a")
    assert set(df5.itertuples(index=False, name=None)) == set(df1.itertuples(index=False, name=None))

    df1 = pd.DataFrame({"a":list("ABCDEF"),
                        "b":range(6),
                        "c":range(6)})
    df2 = pd.DataFrame({"a":list("ACF"),
                        "b":[5,5,5]})
    
    with pytest.raises(Exception):
        fs.override_dataframe(df1, df2)

