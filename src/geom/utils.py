import inspect

__all__ = ["is_single_argument"]


def is_single_argument(fn):
    args = inspect.getfullargspec(fn).args
    kwargs = inspect.getfullargspec(fn).kwonlyargs
    return len(args) == 1 and len(kwargs) == 0
