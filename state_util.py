from equinox.nn import State, StateIndex, make_with_state
import jax


def extract_arg(args, kwargs, arg_name, index: int = -1):
    if arg_name in kwargs:
        state = kwargs[arg_name]
        kwargs = {k: v for k, v in kwargs.items() if k != arg_name}
    else:
        state = args[index]
        args = args[:index] + args[index + 1:]
    return args, kwargs, state


def dummy_caching(fun):
    def fun_(*args, **kwargs):
        args, kwargs, cache = extract_arg(args, kwargs, "cache")
        result = fun(*args, **kwargs)
        if isinstance(result, tuple):
            result = result + (cache,)
        else:
            result = (result, cache)
        return result
    return fun_


def dummy_stateful(fun):
    def fun_(*args, **kwargs):
        args, kwargs, state = extract_arg(args, kwargs, "state")
        result = fun(*args, **kwargs)
        if isinstance(result, tuple):
            result = result + (state,)
        else:
            result = (result, state)
        return result

    return fun_


# copied over from https://github.com/patrick-kidger/equinox/blob/4ea23030e811470f8f947ded52a94c2e28c184cf/equinox/nn/_stateful.py#L362C1-L377C32
def statify(model):
    return make_with_state(lambda: model)()
