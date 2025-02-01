from flask import Flask, request, jsonify
import os
import gspread
from google.oauth2.service_account import Credentials
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import json
import torch


# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Google Sheets Setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("/home/bonnopitom/mysite/ee-csebrurhasinmanjare3434-0f1939796706.json", scopes=SCOPES)
gc = gspread.authorize(creds)
spreadsheet = gc.open("Leads_Management").sheet1

# Load the TinyBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

model = torch.load("/home/bonnopitom/mysite/bert_model.pth")

    


# Define a function to predict label using the TinyBERT model
def predict_label(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get model's prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
    
    # Return the predicted label: 0 -> "Not Interested", 1 -> "Interested"
    return "Interested" if predictions.item() == 1 else "Not Interested"

@app.route('/reply', methods=['POST'])
def handle_reply():
    try:
        # Get the JSON data from SendGrid's webhook
        reply_data = request.json

        # Log full webhook response for debugging
        app.logger.debug(f"Webhook Response: {json.dumps(reply_data, indent=2)}")

        # Ensure the webhook data is a list of events
        if not isinstance(reply_data, list):
            return jsonify({'message': 'Invalid data format'}), 400

        for event in reply_data:
            sender = event.get('email') or event.get('from_email')  # Extract sender email
            body = event.get('text', '').strip().lower()  # Normalize body text

            # Log extracted data
            app.logger.debug(f"Sender: {sender}, Body: {body}")

            # If email is missing, skip processing
            if not sender:
                app.logger.warning("No sender email found, skipping.")
                continue

            # Try to find the email in Google Sheets
            try:
                cell = spreadsheet.find(sender, in_column=2)  # Assuming emails are in column 2
                row = cell.row
            except gspread.exceptions.CellNotFound:
                app.logger.warning(f"Email {sender} not found in Google Sheets.")
                continue  # Skip if email not found

            # Use the model to predict the status based on email body
            status = predict_label(body)

            # Update Google Sheets with the predicted status
            spreadsheet.update_cell(row, 7, status)
            app.logger.info(f"Updated row {row} for {sender} with status '{status}'")

        return jsonify({'message': 'Reply processed successfully!'}), 200

    except Exception as e:
        app.logger.error(f"Error processing the reply data: {e}")
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
