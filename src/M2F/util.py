import re
import os
import warnings
import functools
from typing import Any, Type

def files_from(dir_path: str, pattern: re.Pattern = None):
    pattern = pattern or re.compile(r".*")
    for file_name in sorted(os.listdir(dir_path)):
        if re.match(pattern, file_name):
            yield os.path.join(dir_path, file_name)

def compose(*funcs):
    def inner(x: Any, **fun_args_map):
        for f in funcs:
            args = fun_args_map[f]
            x = f(x, *args)
        return x
    return inner

def suppress_warnings(*warning_types: Type[Warning]):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for w in warning_types:
                    warnings.simplefilter("ignore", w)
                return func(*args, **kwargs)
        return wrapper
    return decorator
