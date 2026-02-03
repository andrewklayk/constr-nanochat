from .tokenizer import get_tokenizer


tok = get_tokenizer()
# print(tok.encode_special('the'))

print(tok.encode('\n'))
print(tok.encode(' '))
# print(tok.decode([tok.get_vocab_size()-13]))
# for i in range(tok.get_vocab_size()):
#     print(tok.id_to_token(i))