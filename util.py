from equinox.nn import make_with_state
import equinox as eqx


def statify(model):
    return make_with_state(lambda: model)()
