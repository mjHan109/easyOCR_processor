import requests
import base64
import cv2 as cv
import time
import sys, os
from glob import glob
import sys


def send_request_to_api(image, api_address, use_cpu=False, use_quad=False):
    image = image[..., ::-1]
    image_bytes = cv.imencode('.jpg', image)[1]
    b64_string = base64.b64encode(image_bytes).decode('utf-8')
    response = requests.post(api_address, json={"image": b64_string})
   
    return response.json()



def send_request_to_ocr(image, api_address, mode, is_group_lines=False, is_bilateral=False, use_cpu=False, use_quad=False):
    if mode == "ocr":    
        endpoint = f"http://{api_address}/ocr_all"
        
        tick = time.perf_counter()
        response = send_request_to_api(image, endpoint, use_cpu=use_cpu, use_quad=use_quad)
        tock = time.perf_counter()
        print(f"Time: {tock - tick:0.4f} seconds")
        words = response['words']
        try:
            if response["result"] == "F":
                print(f"Error: {response['resultDesc']}")
            else:
                print("Success! Do something with the result.")
        except:
            pass
        print(words)       


    elif mode == "status":
        endpoint = f"http://{api_address}/status"
    
        tick = time.perf_counter()
        response = send_request_to_api(image, endpoint, use_cpu=use_cpu, use_quad=use_quad)
        print(response)

def ocr_inference(file_list : list, api_address : str, mode : str):
    for filepath in file_list :                            

        try :
            image = cv.imread(filepath)
            response = send_request_to_ocr(image, api_address, mode, is_group_lines=False, is_bilateral=False, use_cpu=False, use_quad=False)
        except :
            print(filepath)

def collect_imgfiles(path : str, img_format = ['.jpg', '.png']) :
    imgfile_list = []

    for fmt in img_format :
        imgfile_list += glob(os.path.join(path, f'*.{fmt}'))

    return imgfile_list



if __name__ == "__main__":
    path = 'GLS01_노량진1관_스퀘어_냉난방공조_통장사본.jpg'
    port = sys.argv[1]
    api_address = f'0.0.0.0:{port}'
    mode = 'ocr'
    imgfile_list = []

    if os.path.isfile(path) :
        imgfile_list = [path]
    elif os.path.isdir(path) :
        imgfile_list = collect_imgfiles(path)


    ocr_inference(imgfile_list, api_address, mode)
                    
