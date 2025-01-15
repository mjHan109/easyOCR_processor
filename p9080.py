import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
import time
import unicodedata
import uuid
import sys

from fastapi import HTTPException

# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import base64
import cv2 as cv
import numpy as np
import paddleclas
import uvicorn
import asyncio
from fastapi import FastAPI, File, Request, UploadFile
from pydantic import BaseModel
# from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
# from jinja2 import Template
from PIL import Image, ImageDraw, ImageFont

from paddleocr import PaddleOCR
# import pyfiglet
from typing import Optional
import requests

class ImageType(BaseModel):
    image: str = None
    # use_cpu: bool = False
    # use_quad: bool = False


model = paddleclas.PaddleClas(model_name="language_classification")
model_orientation = paddleclas.PaddleClas(model_name="textline_orientation")

IS_VISUALIZE = True
IS_DESKEW = False
IS_BLUR_DETECT = True
# MAX_LEN_SIDE = 1280
# MIN_LEN_SIDE = 960

# MAX_LEN_SIDE = 2560
# MIN_LEN_SIDE = None

# MIN_LEN_SIDE = 768
# MAX_LEN_SIDE = None

MIN_LEN_SIDE = 736
MAX_LEN_SIDE = 2560

# MIN_LEN_SIDE = None
# MAX_LEN_SIDE = None

dir_log = 'output'
if not os.path.exists(dir_log):
    os.makedirs(dir_log)

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory=dir_log), name=dir_log)

# use preprocess of DB

ocr = PaddleOCR(
    # rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.6_google',
    rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.8_dml/Teacher',
    max_text_length=25,
    rec_char_dict_path='ppocr/utils/dict/koreadeep_char_dict_unicode.txt',
    # rec_char_dict_path='ppocr/utils/dict/koreadeep_token_dict.txt',
    # rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.4_hanja_25_syl',
    # max_text_length=25,
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_dml_2.1.2_lite/Teacher',
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_cml_2.1.4_lite/Teacher',
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_dml_3.0.1/Student',
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_cml_3.0.0_lite/Student2',
    # det_algorithm='DB',
    # use_gpu=False,
    use_gpu=True,
    use_dilation=False,
    det_limit_side_len=5,
    det_limit_type='min',
    use_space_char=True,
    use_mp=True,
    # det_box_type='quad',
    det_box_type='poly',
    det_db_score_mode='slow',
    det_db_unclip_ratio=2.7,
    det_max_candidates=5000,
    save_crop_res=False,
    drop_score=0.4,
    # det_limit_side_len=960,
    # det_limit_type='max',
    # det_db_thresh=0.25,
    # det_db_box_thresh=0.5,
)
cpu_ocr = PaddleOCR(
    # rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.6_google',
    rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.8_dml/Teacher',
    max_text_length=25,
    rec_char_dict_path='ppocr/utils/dict/koreadeep_char_dict_unicode.txt',
    # rec_char_dict_path='ppocr/utils/dict/koreadeep_token_dict.txt',
    # rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.4_hanja_25_syl',
    # max_text_length=25,
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_dml_2.1.2_lite/Teacher',
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_cml_2.1.4_lite/Teacher',
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_dml_3.0.1/Student',
    # det_model_dir='inference/det/koreadeep_PP-OCRv3_det_cml_3.0.0_lite/Student2',
    # det_algorithm='DB',
    # use_gpu=False,
    use_gpu=False,
    use_dilation=False,
    det_limit_side_len=5,
    det_limit_type='min',
    use_space_char=True,
    use_mp=True,
    # det_box_type='quad',
    det_box_type='poly',
    det_db_score_mode='slow',
    det_db_unclip_ratio=2.7,
    det_max_candidates=5000,
    save_crop_res=False,
    drop_score=0.4,
    # det_limit_side_len=960,
    # det_limit_type='max',
    # det_db_thresh=0.25,
    # det_db_box_thresh=0.5,
)
quad_poly = PaddleOCR(
    rec_model_dir='inference/rec/koreadeep_PPOCRv3_5.0.8_dml/Teacher',
    max_text_length=25,
    rec_char_dict_path='ppocr/utils/dict/koreadeep_char_dict_unicode.txt',
    use_gpu=True,
    use_dilation=False,
    det_limit_side_len=5,
    det_limit_type='min',
    use_space_char=True,
    use_mp=True,
    det_box_type='quad',
    # det_box_type='poly',
    det_db_score_mode='slow',
    det_db_unclip_ratio=2.7,
    det_max_candidates=5000,
    save_crop_res=False,
    drop_score=0.4,
)

def get_time():
    return time.strftime("%Y%m%d"),time.strftime("%H%M%S")

def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv.imdecode(np_arr, cv.IMREAD_COLOR)


def resize_normalize(img, max_side_len=MAX_LEN_SIDE, min_side_len=MIN_LEN_SIDE):
    h, w, _ = img.shape

    if max_side_len and max(h, w) > max_side_len:
        scale_max = max_side_len / max(h, w)
        img = cv.resize(img, None, fx=scale_max, fy=scale_max, interpolation=cv.INTER_LINEAR_EXACT)
        h, w, _ = img.shape
    else:
        scale_max = 1

    if min_side_len and min(h, w) < min_side_len:
        scale_min = min_side_len / min(h, w)
        img = cv.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv.INTER_CUBIC)
    else:
        scale_min = 1

    scale = scale_max * scale_min
    # crop so that both shape are multiples of 32
    h, w, _ = img.shape
    h = int(h // 32 * 32)
    w = int(w // 32 * 32)
    img = cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR_EXACT)
    return img, scale


def get_deskew_angle(image, result):
    def rotate_box(box, angle):
        box = np.array(box, dtype=np.float32)
        center = np.mean(box, axis=0)
        box -= center
        box = np.dot(box, np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]]))
        box += center
        return box

    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    for res in result:
        for line in res:
            box = np.array(line[0], dtype=np.int32)
            cv.fillPoly(mask, [box], 0)

    angle = get_angle(mask)
    output_image = rotate(image, angle)

    # Rotate all boxes
    for res in result:
        for line in res:
            box = np.array(line[0], dtype=np.int32)
            box = rotate_box(box, angle)
            line[0] = box
    cv.imwrite('mask.jpg', mask)
    return output_image, result

def prepare_ocr(img_path):
    img = cv.imread(img_path)
    img, scale = resize_normalize(img, max_side_len=MAX_LEN_SIDE, min_side_len=MIN_LEN_SIDE)
    result = ocr.ocr(img, cls=True, det=True, rec=True)
    if len(result)>=0:
        print("Success to load OCR")
    else:
        print("Faield to load OCR")
prepare_ocr('sample_image.jpg')

# Recognize small box
@app.post("/ocr_recognize")
async def ocr_recognize(result:ImageType):
#async def ocr_recognize(image: UploadFile=File(...)):
    image = result.image
    use_cpu = result.use_cpu
    tick = time.perf_counter()
    YMD, HMS = get_time()
    request_id = YMD + '-' + HMS
    request_id = f'rec_{request_id}'
    print(f'[{request_id}] Start ocr_recognize')

    #image_bytes = await image.read()
    image_bytes = base64.b64decode(image)
    image = img_decode(image_bytes)
    result,time_dict = ocr.ocr(image, cls=False, det=False)

    tock = time.perf_counter()

    response = {}
    response['request_id'] = request_id
    response['process_time'] = f'{tock - tick:.2f}'
    response['txt'] = result[0][0][0]
    response['score'] = result[0][0][1]

    return response


@app.post("/ocr_detect_paddle")
async def ocr_detect_paddle(result:ImageType):
#async def ocr_detect_paddle(request: Request, image: UploadFile=File(...), use_cpu=False):
    image = result.image
    use_cpu = result.use_cpu
    tick = time.perf_counter()
    YMD, HMS = get_time()
    request_id = YMD + '-' + HMS
    request_id = f'det_{request_id}'
    print(f'[{request_id}] Start ocr_detect_paddle')
    image_bytes = base64.b64decode(image)
    image = img_decode(image_bytes)

    image, scale = resize_normalize(image, max_side_len=MAX_LEN_SIDE, min_side_len=MIN_LEN_SIDE)

    if IS_DESKEW:
        image = get_deskew_angle(image)

    if use_cpu:
        result,time_dict = cpu_ocr.ocr(image, cls=False, det=True, rec=False, is_blur_detect=IS_BLUR_DETECT)
    else:
        result,time_dict = ocr.ocr(image, cls=False, det=True, rec=False, is_blur_detect=IS_BLUR_DETECT)
    tock_model = time.perf_counter()

    print(f'Model process time: {tock_model - tick:.2f}')

    list_words = []

    for res in result:
        for line in res:
            boxes = np.array(line, np.int)
            list_words.append({'box': boxes.tolist(), 'txt': ''})

    tock = time.perf_counter()

    if IS_VISUALIZE:
        dir_output = os.path.join(dir_log, request_id)
        os.makedirs(dir_output, exist_ok=True)

        cv.imwrite(f'{dir_output}/0_input.jpg', image)

        for word in list_words:
            cv.polylines(image, [np.array(word['box'], np.int)], True, (53, 81, 92), 2)
        cv.imwrite(f'{dir_output}/1_detect.jpg', image)

    response = {}
    response['request_id'] = request_id
    response['image_width'] = image.shape[1]
    response['image_height'] = image.shape[0]
    response['original_width'] = int(image.shape[1] / scale)
    response['original_height'] = int(image.shape[0] / scale)
    response['process_time'] = f'{tock - tick:.2f}'
    if IS_VISUALIZE:
        #response['log_image'] = f'http://{request.client.host}:{request.url.port}/log_api/{YMD}/{HMS}/1_detect.jpg'
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        img_base64 = base64.b64encode(cv.imencode('.jpg', image)[1]).decode('utf-8')
        response['image'] = img_base64
    response['words'] = list_words
    return response


async def draw_crop_rec_res(output_dir,ocr_list, thresh=1):
    for ocr_index, result in enumerate(ocr_list):
        img_crop_list , rec_res = result
        bbox_num = len(img_crop_list)
        f = open(os.path.join(output_dir, f"rec_res{ocr_index}.txt"), "w")
        for bno in range(bbox_num):
            if rec_res[bno][1] > thresh:
                continue
            output_path = os.path.join(
                output_dir,
                f"crop{ocr_index}_{bno}.jpg"
            )
            cv.imwrite(output_path, img_crop_list[bno])
            print(f"crop{ocr_index}_{bno}.jpg\t{rec_res[bno][0]}\t{rec_res[bno][1]:.3f}", file=f)
        f.close()

class OCRResult(BaseModel):
    result: str
    resultCode: int
    resultDesc: str
    content: Optional[dict] = None


@app.post("/ocr_all")
async def ocr_all(result:ImageType): 
    try:
        if not result.image:
            response_data = OCRResult(
            result="F",  
            resultCode="444",  
            resultDesc="이미지가 없습니다."
        )
            return response_data
        

        ocr_server_url = "http://127.0.0.1:5112/server-status"

        updated_status_data = {
            '9080_status': True
        }
        requests.post(ocr_server_url, json=updated_status_data)


        YMD, HMS = get_time()
        image = result.image
        use_cpu = False
        use_quad = False
        # use_cpu = result.use_cpu
        # use_quad = result.use_quad
        request_id = YMD + '-' + HMS
        print(f'[{request_id}] Start OCR_all')

        tick = time.perf_counter()
        image_bytes = base64.b64decode(image)
        image = img_decode(image_bytes)

        image, scale = resize_normalize(image, max_side_len=MAX_LEN_SIDE, min_side_len=MIN_LEN_SIDE)

        if use_cpu:
            result, time_dict, org_ocr_res = cpu_ocr.ocr(image, cls=False, det=True, rec=True)
        else:
            if use_quad:
                result, time_dict, org_ocr_res = quad_poly.ocr(image, cls=False, det=True, rec=True)
            else:  
                result, time_dict, org_ocr_res = ocr.ocr(image, cls=False, det=True, rec=True)


        output_dir = os.path.join("output", YMD)
        if not os.path.exists(output_dir):
            image_index = 0
        else:
            image_index = len(os.listdir(output_dir))
        output_dir = os.path.join(output_dir, f"image{image_index}")
        os.makedirs(output_dir, exist_ok=True)
        asyncio.gather(
                draw_crop_rec_res(
                    output_dir,
                    org_ocr_res,
                    thresh=0.98))
        print(time_dict)

        tock_model = time.perf_counter()

        print(f'Model process time: {tock_model - tick:.2f}')

        # list_output = []

        # for res in result_det:
        #     for line in res:
        #         box = np.array(line, 'int')
        #         # print(box)
        #         img_crop = image[box[1][1]:box[2][1], box[0][0]:box[1][0]]
        #         if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
        #             continue
        #         result_rec = ocr.ocr(img_crop, cls=False, det=False, rec=True)
        #         # print(result_rec)
        #         txt = result_rec[0][0][0]
        #         score = result_rec[0][0][1]

        #         result_rec_2 = ocr_chinese.ocr(img_crop, cls=False, det=False, rec=True)
        #         # print(result_rec_2)
        #         txt_2 = result_rec_2[0][0][0]
        #         score_2 = result_rec_2[0][0][1]

        #         if score_2 > 0.8 or True:
        #             txt = txt_2
        #             score = score_2

        #         list_output.append({
        #             'txt': txt,
        #             'score': score,
        #             'box': box.tolist()
        #         })

        if IS_DESKEW:
            image, result = get_deskew_angle(image, result)

        list_output = []
        mean_score = 0
        for res in result:
            for word in res:
                box = np.array(word[0], 'int').tolist()
                txt = word[1][0]
                score = word[1][1]
                mean_score += score
                list_output.append({
                    'txt': txt,
                    # 'score': score,
                    # 'box': box,
                })
        # mean_score /= len(list_output)
        tock = time.perf_counter()

        # if IS_VISUALIZE:

        #     cv.imwrite(f'{output_dir}/input.jpg', image)

            # image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # img_pil = Image.fromarray(image)
            # draw = ImageDraw.Draw(img_pil)
            # font = ImageFont.truetype('NotoSansKR-Regular.otf', 15)

            # for word in list_output:
            #     box = [tuple(b) for b in word['box']]

            #     index_x_min = np.argmin([b[0] for b in box])
            #     box_text_pos = box[index_x_min]

            #     txt = word['txt']
            #     txt = unicodedata.normalize('NFKC', txt)
            #     draw.polygon(box, outline=(255, 0, 0), width=2)
            #     # draw.text(box[0], txt, font=font, fill=(255, 0, 0))
            #     draw.text(
            #         [box_text_pos[0], box_text_pos[1] - 20],
            #         txt,
            #         font=font,
            #         fill=(255, 0, 0))

            # image_rgb = np.array(img_pil)
            # image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
            # cv.imwrite(f'{output_dir}/predict.jpg', image)
            # img_pil.save(f'{dir_output}/1_detect.jpg')
        # image to base64
        if not IS_VISUALIZE:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        img_base64 = base64.b64encode(cv.imencode('.jpg', image)[1]).decode('utf-8')

        response = {}
        response['request_id'] = request_id
        # response['image_width'] = image.shape[1]
        # response['image_height'] = image.shape[0]
        # response['original_width'] = int(image.shape[1] / scale)
        # response['original_height'] = int(image.shape[0] / scale)
        response['process_time'] = f'{tock - tick:.2f}'
        # if IS_VISUALIZE:
        #     # response[
        #     #     'log_image'] = f'http://{request.client.host}:{request.url.port}/{output_dir}/predict.jpg'
        #     response['image'] = img_base64

        # words = words + "\n"
        str_txt = ""

        updated_status_data = {
            '9080_status': False
        }
        requests.post(ocr_server_url, json=updated_status_data)

        for word in list_output:
            str_txt += word['txt'] + "\n"
            
        response['words'] = str_txt
        print(str_txt)
        return response

    except HTTPException as e:
        # HTTPException이 발생한 경우 (클라이언트로부터의 잘못된 요청 등)
        response_data = OCRResult(
            result="F",  # 실패한 경우 "F"로 설정
            resultCode=e.status_code,  # 실패 상태 코드 설정
            resultDesc=str(e.detail)
        )
        return response_data
    
    except Exception as e:
        # 그 외의 예외가 발생한 경우
        error_message = f"요청 처리 중 오류 발생: {str(e)}"
        response_data = OCRResult(
            result="F",  # 실패한 경우 "F"로 설정
            resultCode=500,  # 실패 상태 코드 설정 (예시로 500 사용)
            resultDesc=error_message
        )
        return response_data



@app.post("/img_language")
async def img_language(request: Request, image: UploadFile=File(...)):
    YMD, HMS = get_time()
    request_id = f'lang_{YMD}-{HMS}'
    print(f'[{request_id}] Start img_language')

    tick = time.perf_counter()
    image_bytes = await image.read()
    image = img_decode(image_bytes)

    # otsu
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, img_binary = cv.threshold(img_gray, 0, 255,
                                 cv.THRESH_BINARY + cv.THRESH_OTSU)
    image = cv.cvtColor(img_binary, cv.COLOR_GRAY2BGR)

    result = model.predict(input_data=image)

    tock = time.perf_counter()

    response = {}
    response['request_id'] = request_id
    response['process_time'] = f'{tock - tick:.2f}'
    response['result'] = next(result)
    return response


@app.post("/img_orientation")
async def img_orientation(request: Request, image: UploadFile=File(...)):
    YMS, HMS = get_time()
    request_id = f'orient_{YMS}-{HMS}'
    print(f'[{request_id}] Start img_orientation')

    tick = time.perf_counter()
    image_bytes = await image.read()
    image = img_decode(image_bytes)

    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, img_binary = cv.threshold(img_gray, 0, 255,
                                 cv.THRESH_BINARY + cv.THRESH_OTSU)
    image = cv.cvtColor(img_binary, cv.COLOR_GRAY2BGR)

    result = model_orientation.predict(input_data=image)

    tock = time.perf_counter()

    response = {}
    response['request_id'] = request_id
    response['process_time'] = f'{tock - tick:.2f}'
    response['result'] = next(result)
    return response


def get_mac_address():
    mac_address = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(5, -1, -1)])
    return mac_address


def run_deepocr() :
    import getmac
    current_mac_address = getmac.get_mac_address()

    allowed_mac_address = "9a:64:3c:ba:dd:b7"
    
    if current_mac_address == allowed_mac_address:
        uvicorn.run(app, host="0.0.0.0", port=9080)
    else:
        print("E: M407")
    

if __name__ == "__main__":
    # ascii_art = pyfiglet.figlet_format("DeepOCRv2")

    # print(ascii_art)
    # user_input = input("Enter Port: ")
    # print("Port: ", user_input)
    uvicorn.run(app, host="0.0.0.0", port=9080)     