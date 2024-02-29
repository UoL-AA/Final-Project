from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request
import torch

# Load pre-trained BERT model and tokenizer
loaded_tokenizer = BertTokenizer.from_pretrained('bert_model')
loaded_bert_model = BertForSequenceClassification.from_pretrained('bert_model', num_labels=2)

# Start Flask
app = Flask(__name__)

# Function to get the result for a particular text query
def request_results(text_input):
    # Tokenize the input text
    tokens = loaded_tokenizer.tokenize(loaded_tokenizer.decode(loaded_tokenizer.encode(text_input, add_special_tokens=True)))

    # Truncate or pad the tokens to the specified max length
    max_len = 128
    tokens = tokens[:max_len - 2] + [loaded_tokenizer.pad_token] * max(0, max_len - 2 - len(tokens))

    # Convert tokens to input IDs
    input_ids = loaded_tokenizer.convert_tokens_to_ids(tokens)

    # Create attention mask
    attention_mask = [1] * len(input_ids)

    # Convert input to PyTorch tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = loaded_bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get predicted class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()

    # Determine result
    if probabilities[0] > probabilities[1]:
        result = "reliable"
        confidence = probabilities[0]
    else:
        result = "unreliable"
        confidence = probabilities[1]

    return {
        "result": result,
        "confidence": round((confidence * 100), 2)
    }

# Render default webpage
@app.route('/')
def home():
    return render_template('home.html')

# When post method is detected, redirect to success function
@app.route('/', methods=['POST'])
def get_data():
    if request.method == 'POST':
        user_input = request.form['search']
        result = request_results(user_input)
        return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
