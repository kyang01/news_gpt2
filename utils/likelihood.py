import torch
import torch.nn as nn

def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar. 
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """
    length = text.shape[1]
    log_likelihood = 0
    with torch.no_grad():
        logits, past = model(text, past=None)
        for i in range(length):
            log_probs = torch.squeeze(torch.nn.functional.log_softmax(logits[:, i, :], dim=1))
            log_likelihood += log_probs[text[0, i]]
    return log_likelihood

        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`