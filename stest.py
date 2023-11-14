class Obj:
    def u(self, x):
        x = (lambda a: a)(x)
        x = (lambda a: a)(x)
        return x

    def f(self, x):
        print(state)
        x = (lambda a: a)(x)
        print(state)
        x = (lambda a: a)(x)
        print(state)
        x = self.u(x)
        print(x, state)
        return x
