from common import *
import pandas as pd
import datetime


def test_repr_invalid_name(df):
    df['is'] = df.x * 1
    code = df._repr_mimebundle_()['text/plain']
    assert '_is' not in code, "the repr should show the aliased name"


def test_repr_default(df):
    code = df._repr_mimebundle_()['text/plain']
    assert 'x' in code


def test_repr_html(df):
    ds = df
    code = ds._repr_html_()
    assert 'x' in code


def test_repr_empty(df):
    df = df[df.x < 0]
    bundle = df._repr_mimebundle_()
    assert 'no rows' in bundle['text/plain'].lower()
    assert 'no rows' in bundle['text/html'].lower()


# TODO: it seems masked arrays + evaluate doesn't work well
# might have to do something with serializing it
def test_mask(df_local):
    ds = df_local
    code = ds._repr_html_()
    assert "'--'" not in code
    assert "--" in code

    code = ds._repr_mimebundle_()['text/plain']
    assert "'--'" not in code
    assert "--" in code


def test_repr_expression(df):
    df = df
    assert 'Error' not in repr(df.x)


def test_repr_df_long_string():
    long_string = "Hi there" * 100
    df = vaex.from_arrays(s=[long_string] * 100)
    assert long_string not in repr(df)
    assert long_string not in str(df)
    assert long_string not in df._repr_html_()
    assert long_string not in df._as_html_table(0, 10)

    # as objects
    df = vaex.from_arrays(o=[{"something": long_string}] * 100)
    assert long_string not in repr(df)
    assert long_string not in str(df)
    assert long_string not in df._repr_html_()
    assert long_string not in df._as_html_table(0, 10)


# TODO: because remote slicing of filtered datasets is not supported, we have a workaround
# we RMI the __repr__
def test_slice_filtered_remte(ds_remote):
    df = ds_remote
    dff = df[df.x > 0]
    assert "0.0bla" not in repr(dff[['x']])


def test_repr_from_pandas():
    dd_dict = {
        'boolean': [True, True, False, None, True],
        'text': ['This', 'is', 'some', 'text', 'so...'],
        'text_missing': pd.Series(['Some', 'parts', None, 'missing', None], dtype='string'),
        'float': [1, 30, -2, 1.5, 0.000],
        'float_missing': [1, None, -2, 1.5, 0.000],
        'int_missing': pd.Series([1, None, 5, 1, 10], dtype='Int64'),
        'datetime_1': [pd.NaT, datetime.datetime(2019, 1, 1, 1, 1, 1), datetime.datetime(2019, 1, 1, 1, 1, 1), datetime.datetime(2019, 1, 1, 1, 1, 1), datetime.datetime(2019, 1, 1, 1, 1, 1)],
        'datetime_2': [pd.NaT, None, pd.NaT, pd.NaT, pd.NaT],
        'datetime_3': [pd.Timedelta('1M'), pd.Timedelta('1D'), pd.Timedelta('100M'), pd.Timedelta('2D'), pd.Timedelta('1H')],
        'datetime_4': [pd.Timestamp('2001-1-1 2:2:11'), pd.Timestamp('2001-12'), pd.Timestamp('2001-10-1'), pd.Timestamp('2001-03-1 2:2:11'), pd.Timestamp('2001-1-1 2:2:11')],
        'datetime_5': [datetime.date(2010, 1, 1), datetime.date(2010, 1, 1), datetime.date(2010, 1, 1), datetime.date(2010, 1, 1), datetime.date(2010, 1, 1)],
        'datetime_6': [datetime.time(21, 1, 1), datetime.time(21, 1, 1), datetime.time(21, 1, 1), datetime.time(21, 1, 1), datetime.time(21, 1, 1)],
    }

    # Get pandas dataframe
    pandas_df = pd.DataFrame(dd_dict)
    pandas_df['datetime_7'] = pd.to_timedelta(pandas_df['datetime_2'] - pandas_df['datetime_1'])
    vaex_df = vaex.from_pandas(pandas_df)
    repr_value = repr(vaex_df)
    str_value = str(vaex_df)

    assert 'NaT' in repr_value
    assert 'NaT' in str_value
    assert '--' in repr_value
    assert '--' in str_value

    assert vaex_df.text_missing.is_masked == True
    assert vaex_df.int_missing.is_masked == True
    assert vaex_df.float_missing.is_masked == False
    assert vaex_df.int_missing.tolist() == [1, None, 5, 1, 10]
    assert vaex_df.text_missing.tolist() == ['Some', 'parts', None, 'missing', None]
    assert vaex_df.float_missing.values[[0, 2, 3, 4]].tolist() == [1.0, -2.0, 1.5, 0.0]
    assert np.isnan(vaex_df.float_missing.values[1])
