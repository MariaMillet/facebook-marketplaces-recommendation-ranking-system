import requests
files = {'image': open('/Users/mariakosyuchenko/AI_Core/facebook-marketplaces-recommendation-ranking-system/cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg','rb')}
# response = requests.post('http://3.250.17.102:8080/predict/image', files={'image':'/Users/mariakosyuchenko/AI_Core/facebook-marketplaces-recommendation-ranking-system/cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg'})
r = requests.post('http://3.250.17.102:8080/predict/image', files=files)
print(r.json())
