import sys
sys.dont_write_bytecode = True

import requests

def get_request(url: str|bytes, headers=None, stream: bool=None):
    # Define the custom User-Agent
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
        }
    response = requests.get(url, headers=headers, stream=stream)
    #response.raise_for_status()  # Raise an exception for HTTP errors
    return response



def post_request(url, headers=None, stream: bool=None, files=None, data=None):
    # Define the custom User-Agent
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
            "CF-Access-Client-Id": "aeb67eaf8fc9dc4757e889f87a2421e1.access",
            "CF-Access-Client-Secret": "600dd9a9585ec842ffd0bc8306cfc28c9958f6d83954c4d51a02acde32520df8",
        }
    response = requests.post(url, headers=headers, stream=stream, files=files, data=data)
    #response.raise_for_status()  # Raise an exception for HTTP errors
    return response


