import vaex


class DataFrameAccessorDistributed:
    def __init__(self, df):
        self.df = df

    def to_dask(self):
        from .dask import executor
        ddf = vaex.dataframe.DataFrameLocal(self.df.dataset)
        ddf.state_set(self.df.state_get())  # transfer virtual columns set
        ddf.executor = executor()
        return ddf

    def to_ray(self):
        from .ray import executor
        ddf = vaex.dataframe.DataFrameLocal(self.df.dataset)
        ddf.state_set(self.df.state_get())  # transfer virtual columns set
        ddf.executor = executor()
        return ddf
