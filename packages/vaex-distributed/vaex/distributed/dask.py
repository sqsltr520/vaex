
from dask import delayed

import vaex
from vaex.encoding import Encoding


@delayed
def do_chunk(df, task_specs, i1, i2):
    print("chunk", i1, i2, task_specs)
    encoding = Encoding()
    task_parts = encoding.decode_list('task-part-cpu', task_specs, df=df)
    all_expressions = list(set(expression for task_part in task_parts for expression in task_part.expressions))
    all_values = {expression: df.columns[expression][i1:i2] for expression in all_expressions}
    for task_part in task_parts:
        values = [all_values[expression] for expression in task_part.expressions]
        task_part.process(0, i1, i2, None, *values)
    return task_parts


@delayed
def reduce(task_parts1, task_parts2):
    print("reduce", task_parts1, task_parts2)
    for task_part1, task_part2 in zip(task_parts1, task_parts2):
        task_part1.reduce([task_part2])
    return task_parts1


class Executor:
    def __init__(self, chunk_size=1_000_000):
        self.tasks = []
        self.chunk_size = chunk_size
        self.server = None

    def schedule(self, task):
        self.tasks.append(task)

    def execute(self, delay=False):
        tasks = list(self.tasks)
        for task in tasks:
            dfs = set(task.df for task in tasks)
            for df in dfs:
                chunk_count = (len(df) + self.chunk_size - 1) // self.chunk_size
                tasks_df = [task for task in tasks if task.df is df]
                task_parts_previous = None
                encoding = Encoding()
                task_specs = encoding.encode_list('task', tasks_df)
                for chunk_index in range(chunk_count):
                    i1 = chunk_index * self.chunk_size
                    i2 = min(len(df), (chunk_index + 1) * self.chunk_size)
                    task_parts = do_chunk(df, task_specs, i1, i2)
                    if task_parts_previous is not None:
                        task_parts = reduce(task_parts_previous, task_parts)
                    task_parts_previous = task_parts
                if delay:
                    return task_parts
                for task, task_parts in zip(tasks_df, task_parts.compute()):
                    task.result = task_parts.get_result()
                    task.fulfill(task.result)
