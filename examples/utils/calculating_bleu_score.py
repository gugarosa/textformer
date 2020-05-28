from torchtext.data.metrics import bleu_score

# Defining a list of target tokens
# Note that when defining targets, you need to use an iterable of iterables of tokens
targets = [[['One', 'can', 'use', 'several', 'sentences']], [['Like', 'this', 'example']]]

# Defining a list of predicted tokens
# Note that when defining predictions, you need to use an iterable of tokens
preds = [['You', 'can', 'use', 'several', 'sentences'], ['Like', 'this', 'example']]

# Calculating BLEU score
bleu = bleu_score(preds, targets)

print(bleu)
