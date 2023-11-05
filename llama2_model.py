import equinox as eqx
import jax


class Embedding(eqx.Module):
    weight: jax.Array
    
    vocab_size: int = 32_000
    hidden_size: int
    is_unembed: bool
    
    def __init__(self, hidden_size, key, is_unembed=False):
        self.hidden_size = hidden_size
        self.is_unembed = is_unembed
        self.weight = (jax.random.normal(
            key,
            (self.vocab_size, self.hidden_size)[::(-1 if is_unembed else 1)])
                       * ((1 / self.hidden_size) ** 0.5))
    
    def __call__(self, x):
        if self.is_unembed:
            return x @ self.weight
        else:
            return self.weight[x]


class LLaMAModel(eqx.Module):
    embed_tokens: Embedding
    hidden_size: int
    
    def __init__(self, hidden_size, key):
        self.hidden_size = hidden_size
        embed_key, key = jax.random.split(key)
        self.embed_tokens = Embedding(self.hidden_size, embed_key)
    
    def __call__(self, x):
        x = self.embed_tokens(x)
        return x


class LLaMA(eqx.Module):
    lm_head: Embedding
    model: LLaMAModel
    
    hidden_size: int = 4096
    
    def __init__(self, key):
        lm_head_key, key = jax.random.split(key)
        self.lm_head = Embedding(self.hidden_size, key=lm_head_key, is_unembed=True)
        self.model = LLaMAModel(self.hidden_size, key)
    
    def __call__(self, x):
        return self.lm_head(self.model(x))
