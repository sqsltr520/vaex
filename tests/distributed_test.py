import pytest
import vaex
import vaex.distributed.dask

# we don't test ray yet with general tasks, because they don't pickle well
if True:  # vaex.utils.devmode:
    @pytest.fixture(params=['ddf_dask'])
    def ddf(request, ddf_dask):
        named = dict(ddf_dask=ddf_dask)
        return named[request.param]
else:
    # ray takes a while to spin up/down
    @pytest.fixture(params=['ddf_dask', 'ddf_ray'])
    def ddf(request, ddf_dask, ddf_ray):
        named = dict(ddf_dask=ddf_dask, ddf_ray=ddf_ray)
        return named[request.param]

@pytest.fixture()
def ddf_dask(df_trimmed):
    return df_trimmed.distributed.to_dask()


@pytest.fixture()
def ddf_ray(df_trimmed):
    return df_trimmed.distributed.to_ray()


def test_distributed_basics(ddf, df_trimmed):
    assert ddf.x.sum() == df_trimmed.x.sum()
    assert ddf.x.mean() == df_trimmed.x.mean()
    assert ddf.x.max() == df_trimmed.x.max()
    assert (ddf.x**2).sum() == (df_trimmed.x**2).sum()
