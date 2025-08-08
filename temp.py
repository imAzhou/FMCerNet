import requests, certifi
print(requests.utils.DEFAULT_CA_BUNDLE_PATH)
print(certifi.where())