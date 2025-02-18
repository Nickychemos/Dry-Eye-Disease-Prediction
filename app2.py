from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests

app = Flask(__name__)


model = joblib.load('model.pkl')

# Meta API credentials
ACCESS_TOKEN = "EAAIqyN9RwAUBO9zIvly7ZAVh0cZA2vjZCwMrZA6Oz0EFjbSHPBJ0lVdZBNtSJeOi2kcuMlEZBjh7o7BUzTHs3VZBwZCvyWg3fVZBpbqjJsS85nYdUHuiEULRcDViMFeItnrTZBBZB7qMDPP2GXEd93WElZCxJEcoHWsYD0cxIQwmB237VXIwkhgydHGou5qarrQFQFr2PEla2d9R2GRP5ImlJ4Y481MRITsZD"
PHONE_NUMBER_ID = "512115608660839"

def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    requests.post(url, headers=headers, json=data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame([[
        data['Eye_Itchiness'],
        data['Eye_Redness'],
        data['Eye_Strain'],
        data['Screen_Time']
    ]], columns=['Eye_Itchiness', 'Eye_Redness', 'Eye_Strain', 'Screen_Time'])
    
    prediction = model.predict(features)[0]
    condition_status = "has Dry Eye Disease" if prediction == 1 else "does not have Dry Eye Disease"
    
    # Send response to WhatsApp
    send_whatsapp_message(data['whatsapp_number'], f"The patient {condition_status}.")
    
    return jsonify({'status': condition_status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
