import requests
url = 'http://127.0.0.1:5000/predict'
files = {'photo_file': open('static/uploads/test_phone.jpg','rb')}
data = {'mode': 'photo'}
resp = requests.post(url, files=files, data=data, timeout=20)
print('Status:', resp.status_code)
print(resp.text)
