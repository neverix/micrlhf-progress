from equinox.nn import make_with_state


def statify(model):
    return make_with_state(lambda: model)()
