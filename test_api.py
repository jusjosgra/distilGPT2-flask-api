import requests
import json

url = 'http://localhost:5000/predict_text'
HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}
data = json.dumps(
{
"text":"It was a dark and stormzy night"
}
)

print(data)

r = requests.post(url = url, data = data, headers = HEADERS)
print(r.text)

