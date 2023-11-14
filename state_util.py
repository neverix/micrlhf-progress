from equinox.nn import State, StateIndex, make_with_state
import jax


def extract_state(args, kwargs):
    if "state" in kwargs:
        state = kwargs["state"]
        kwargs = {k: v for k, v in kwargs.items() if k != "state"}
    else:
        state = args[-1]
        args = args[:-1]
    return args, kwargs, state


def dummy_stateful(fun):
    def fun_(*args, **kwargs):
        args, kwargs, state = extract_state(args, kwargs)
        return fun(*args, **kwargs), state

    return fun_


# copied over from https://github.com/patrick-kidger/equinox/blob/4ea23030e811470f8f947ded52a94c2e28c184cf/equinox/nn/_stateful.py#L362C1-L377C32
def statify(model):
    return make_with_state(lambda: model)()
