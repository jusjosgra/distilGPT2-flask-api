import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

def predict(text, pred_len=20):
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    for i in range(pred_len):
        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        # get the predicted next sub-word (in our case, the word 'man')
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        indexed_tokens = indexed_tokens + [predicted_index]
        tokens_tensor = torch.tensor([indexed_tokens])

    predicted_text = tokenizer.decode(indexed_tokens)
    return predicted_text

if __name__ == "__main__":

    text = "It was a dark and stormy night and the air was filled with"
    predicted_text = predict(text)
    print(predicted_text)

