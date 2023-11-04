from tokenizer import Tokenizer


tokenizer = Tokenizer("models/Llama-2-7b-hf/tokenizer.model")
print(tokenizer.encode("Hello world!", bos=True, eos=False))