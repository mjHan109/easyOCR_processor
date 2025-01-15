# -*- coding: utf-8 -*-

import json
import os
import requests
import base64
import cv2 as cv
import time
import sys
import threading
import logging
import log_info
import concurrent.futures
from concurrent.futures import as_completed
from logging.handlers import RotatingFileHandler
import datetime
import uuid
import re
import shutil
import magic
from PIL import Image, ExifTags
import numpy as np
import main_utility as main_ut


Image.MAX_IMAGE_PIXELS = None
success_cnt = 0
failed_cnt = 0
small_image_cnt = 0
big_image_cnt = 0
json_failed_cnt = 0
ext_cnt = 0
count = 0
end_flag = False
ext_lock = threading.Lock()
count_lock = threading.Lock()
success_lock = threading.Lock()
failed_lock = threading.Lock()
json_failed_lock = threading.Lock()
small_image_lock = threading.Lock()
big_image_lock = threading.Lock()
file_write_lock = threading.Lock()
ext_file_write_lock = threading.Lock()
write_failed_file_write_lock = threading.Lock()
write_json_failed_file_write_lock = threading.Lock()

current_time = time.strftime("%Y%m%d%H%M%S")



def config_reading():
    try:
        current_directory = os.getcwd()
        config_file = os.path.join(current_directory, 'config.json')
        if os.path.isfile(config_file):
            with open(config_file, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                return json_data        
        else:
            logging.error("config.json 파일을 찾을 수 없습니다.")
            return None
    except Exception as e:
        logging.error(f"cofig.json 파일을 읽는 도중 오류가 발생했습니다 : {str(e)}")
        # logging.debug(f"cofig.json 파일을 읽는 도중 오류가 발생했습니다 : {str(e)}")
        return None


def find_image_files(resource_path, result_path, ocr_filter):
    global ext_cnt
    image_files = []
    total_size = 0

    for root, _, files in os.walk(resource_path):
        for file in files:
            file_size = None
            image_path = os.path.join(root, file)
            main_fold = get_main_fold(resource_path, image_path)
            
            # 파일이 심볼릭 링크가 아니고, 읽기 권한이 있는지 확인
            if not os.path.islink(image_path) and os.access(image_path, os.R_OK):
                try:
                    # GIF와 TIF 파일 체크
                    file_extension = os.path.splitext(file)[1].lower()
                    if file_extension in ['.gif', '.tif', '.tiff']:
                        file_size = os.path.getsize(image_path)
                        increment_ext_count()
                        indexing_ext(main_fold, image_path, result_path, "ocr_0015", file_size)
                        continue
                        
                    if file.lower().endswith(ocr_filter):
                        try:
                            file_size = os.path.getsize(image_path)
                            image_files.append(image_path)
                            total_size += file_size
                        except OSError as e:
                            error_message = str(e)
                            increment_ext_count()
                            indexing_ext(main_fold, image_path, result_path, "ocr_0016", file_size)
                            logging.error(f"파일 크기를 가져오는 중 오류 발생: {image_path}, 오류: {error_message}")
                except Exception as e:
                    error_message = str(e)
                    increment_ext_count()
                    indexing_ext(main_fold, image_path, result_path, "ocr_0016", file_size)
                    logging.error(f"파일 처리 중 예외 발생: {image_path}, 오류: {error_message}")
            elif os.path.islink(image_path):
                file_size = None
                logging.debug(f"심볼릭 링크 파일 : {image_path}")
                increment_ext_count()
                indexing_ext(main_fold, image_path, result_path, "ocr_0016", file_size)
            else:
                file_size = None
                logging.debug(f"파일에 읽기 권한이 없습니다. : {image_path}")
                increment_ext_count()
                indexing_ext(main_fold, image_path, result_path, "ocr_0017", file_size)


    total_size_gb = total_size / (1024**3)
    
    return image_files, total_size_gb


def get_main_fold(target_path, file_path):
    relative_path = os.path.relpath(file_path, target_path)
    components = os.path.normpath(relative_path).split(os.path.sep)
    main_fold = components[0] if len(components) > 1 else None
    
    if main_fold:
        parts = main_fold.split('_')
        filtered_parts = [part for part in parts if not part.isdigit()]
        main_fold = '_'.join(filtered_parts)
    
    return main_fold
 

def increment_success_count():
    global success_cnt
    with success_lock:
        # success_lock.acquire()
        success_cnt += 1
        # success_lock.release()


def increment_failed_count():
    global failed_cnt
    with failed_lock:
        failed_cnt += 1


def increment_json_failed_count():
    global json_failed_cnt
    with json_failed_lock:
        json_failed_cnt += 1


def increment_progress_count():
    global count
    with count_lock:
        count += 1
    return count


def increment_small_size_count():
    global small_image_cnt
    with small_image_lock:
        small_image_cnt += 1
    return small_image_cnt


def increment_big_size_count():
    global big_image_cnt
    with big_image_lock:
        big_image_cnt += 1
    return big_image_cnt


def increment_ext_count():
    global ext_cnt
    with ext_lock:
        ext_cnt += 1
    return ext_cnt




# meta 정보 추출
def read_file_from_path(file_path):
    try:
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        creation_time = time.ctime(file_stats.st_ctime)
        modification_time = time.ctime(file_stats.st_mtime)
        access_time = time.ctime(file_stats.st_atime)
        uid = file_stats.st_uid
        gid = file_stats.st_gid
        owner_info = f"{uid}:{gid}"
        file_size  = file_size / 1024

        meta_info = {
            "accessed": access_time,
            "created":creation_time,
            "mtime": modification_time,
            "size": file_size,
            "owner": owner_info
        }

        return meta_info
    except Exception as e:      
        info_message = f"{file_path}, a exceptions: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return None



def get_log_level(log_level):
    log_level = log_level.upper()
    if log_level == "DEBUG":
        return logging.DEBUG
    elif log_level == "INFO":
        return logging.INFO
    elif log_level == "WARNING":
        return logging.WARNING
    elif log_level == "ERROR":
        return logging.ERROR
    elif log_level == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO



# 진행 상황, 결과 
def count_and_print_ocr_results(success_cnt, failed_cnt, total_cnt, count, json_failed_cnt, ext_cnt, start_time, end_flag):
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if end_flag:
        logging.info(f"[INFO] OCR 분석 결과")
        logging.info(f"총 파일 개수: {total_cnt}")
    else:
        logging.info(f"[INFO] OCR 진행 상황")
        logging.info(f"총 파일 개수: {total_cnt}")
        logging.info(f"진행 상황: {count}/{total_cnt}")

    logging.info(f"OCR 분석 성공 파일 : {success_cnt}개")
    logging.info(f"OCR 분석 실패 파일 : {failed_cnt}개")
    logging.info(f"색인 제외 파일 : {ext_cnt}개")
    logging.info(f"JSON 저장 실패 파일 : {json_failed_cnt}개")
    logging.info(f"OCR 분석 시간: {int(hours)} 시간 {int(minutes)} 분 {int(seconds)} 초")




def  send_request_to_api(idx, image, endpoint, resource_path, file_path, result_path, use_cpu, use_quad):
    global end_flag, success_cnt, failed_cnt, json_failed_cnt
    json_data = config_reading()
    response_data = {}
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    main_fold = get_main_fold(resource_path, file_path)
    flip_flag = None
    file_size = os.path.getsize(file_path)

    try:
        flip_flag = json_data["ocr_info"]["flip_flag"]
    except Exception as e:
        logging.error("flip_flag가 존재하지 않습니다.")

    try:
        # image = image[..., ::-1]
        if flip_flag:
            flipped_image = cv.flip(image, 1)
            image_bytes = cv.imencode('.jpg', flipped_image)[1]
        else:
            status, image_bytes = cv.imencode('.jpg', image)
            if status is not True:
                increment_ext_count()
                indexing_ext(main_fold, file_path, result_path, "ocr_0005", file_size)
                logging.error(f"imencoding failed, File : {file_path}")
        try:
            b64_string = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            error_msg = str(e)
            logging.error(f"File : {file_path}, Base64EncodingError : {error_msg}")
            indexing_ext(main_fold, file_path, result_path, "ocr_0014", file_size)
        
        response = requests.post(endpoint, json={"image": b64_string, "name" : file_name})

        if response is not None:
            if response.status_code == 200:
                response_data = response.json()
                if response_data is not None:
                    if 'words' in response_data:
                        increment_success_count()
                        logging.debug(f"{endpoint}, File : {file_path}, API Response : {response_data}")
                        json_failed_cnt = save_as_json(file_path, result_path, response_data)
                    elif 'resultCode' in response_data:
                        # 분석에 실패했을 때
                        if 500 == response_data['resultCode']:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0013", "exception_message" : "예측하지 못한 에러 발생"}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)
                        elif 501 == response_data['resultCode']:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0013", "exception_message" : "이미지가 없는 경우"}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)
                        elif 502 == response_data['resultCode']:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0013", "exception_message" : "이미지가 base64로 전송되지 않았습니다."}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)
                        elif 503 == response_data['resultCode']:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0013", "exception_message" : "MemoryError 발생"}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)
                        elif 504 == response_data['resultCode']:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0013", "exception_message" : "CUDA out of memory error 발생"}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)
                        elif 505 == response_data['resultCode']:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0013", "exception_message" : "RuntimeError 발생"}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)
                        else:
                            increment_failed_count()
                            response_data = {"tags": ["ocr", "file", "N"], "mime_type" : "ocr_0012", "exception_message" : "알 수 없는 오류 발생"}
                            logging.debug(f"File : {file_path}, API Response : {response_data}")
                            json_failed_cnt = save_as_json(file_path, result_path, response_data)

                else:
                    # response_data가 None일 때
                    if 'resultDesc' in response_data:
                        response_data = {"tags": ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0006", "exception_message" : response_data['resultDesc']}
                    # response_data에 resultDesc가 없을 때
                    else:
                        response_data = {"tags": ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0006", "exception_message" : f"Status Code : {response.status_code}"}
                    increment_failed_count()
                    logging.debug(f"File : {file_path}, response의 json이 None")
                    logging.error(f"File : {file_path}, response의 json이 None")
                    json_failed_cnt = save_as_json(file_path, result_path, response_data)
            else:
                increment_failed_count()
                if response.status_code:
                    # 응답 값에 status_code가 존재할 때
                    response_data = {"tags": ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0010", "exception_message" : f"Status Code : {response.status_code}"}
                else:
                    response_data = {"tags": ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0010", "exception_message" : "이미지 분석에 실패했습니다."}
                logging.debug(f"{endpoint}, File : {file_path}, Status Code is not 200")
                logging.error(f"{endpoint}, File : {file_path}, Status Code is not 200")
                json_failed_cnt = save_as_json(file_path, result_path, response_data)
        else:
            increment_failed_count()
            error_msg = "Response is None"
            response = {"tags": ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0006", "exception_message" : error_msg}
            logging.debug(f"{endpoint}, File : {file_path}, Response is None")
            logging.error(f"{endpoint}, File : {file_path}, Response is None")
            json_failed_cnt = save_as_json(file_path, result_path, response)     
    except requests.exceptions.RequestException as request_exception:
        increment_failed_count()
        error_msg = str(request_exception)
        response = {"tags": ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0011", "exception_message" : error_msg}
        logging.debug(f"{endpoint}, File : {file_path}, RequestException : {error_msg}")
        logging.error(f"{endpoint}, File : {file_path}, RequestException : {error_msg}")
        json_failed_cnt = save_as_json(file_path, result_path, response)
    except json.decoder.JSONDecodeError as e:
        increment_failed_count()
        error_msg = str(e)
        response = {"tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : error_msg}
        # logging.debug(f"File : {file_path}, JSONDecodeError : {response_data}")
        logging.debug(f"{endpoint}, File : {file_path}, JSONDecodeError : {error_msg}")
        logging.error(f"{endpoint}, File : {file_path}, JSONDecodeError : {error_msg}")
        json_failed_cnt = save_as_json(file_path, result_path, response)
    except Exception as e:
        increment_failed_count()
        error_msg = str(e)
        response = {"tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : error_msg}
        # logging.debug(f"File : {file_path}, Exception : {response_data}")
        logging.debug(f"{endpoint}, File : {file_path}, After Response Exception : {error_msg}")
        logging.error(f"{endpoint}, File : {file_path}, After Response Exception : {error_msg}")
        json_failed_cnt = save_as_json(file_path, result_path, response)



def send_request_to_ocr(idx, image, ocr_api, resource_path, file_path, result_path):
    global success_cnt, failed_cnt, json_failed_cnt, ext_cnt
    json_data = config_reading()
    ocr_info = json_data['ocr_info']
    sleep_time = ocr_info['sleep_time']
    is_bilateral = ocr_info['is_bilateral']
    is_group_lines = ocr_info['is_group_lines']
    use_cpu = ocr_info['use_cpu']
    use_quad = ocr_info['use_quad']
    file_size = os.path.getsize(file_path)
    main_fold = get_main_fold(resource_path, file_path)

    if is_bilateral:
        image = cv.bilateralFilter(image, 9, 75, 75)

    endpoint = f"http://{ocr_api}/ocr_all"

    # logging.debug(f"endpoint : {endpoint}")
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    
    # GIF 파일인 경우 처리하지 않음
    if file_extension.lower() == '.gif':
        increment_ext_count()
        logging.info(f"GIF 파일은 처리하지 않습니다: {file_path}")
        indexing_ext(main_fold, file_path, result_path, "ocr_0015", file_size)
        return

    try:
        send_request_to_api(idx, image, endpoint, resource_path, file_path, result_path, use_cpu, use_quad)
    
    except Exception as e:
        increment_failed_count()
        error_msg = str(e)
        response_data = {'tags': ["ocr", "file", "N", "exception"], 'mime_type': 'ocr_0008', 'exception_message': error_msg}
        logging.debug(f"{endpoint}, File : {file_path}, resultDesc : {response_data}")
        logging.error(f"{endpoint}, File : {file_path}, resultDesc : {response_data}")
        json_failed_cnt = save_as_json(file_path, result_path, response_data)
    
    time.sleep(sleep_time)


def rotate_image_if_needed(img):
    try:
        # EXIF 태그 중 Orientation 키를 찾습니다.
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        # 이미지의 EXIF 데이터에서 방향 정보를 가져옵니다.
        exif=dict(img._getexif().items())
        
        # 방향 정보에 따라 이미지를 회전시킵니다.
        if exif[orientation] == 3:
            img=img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img=img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img=img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # EXIF 정보가 없거나 방향 정보가 없는 경우 오류를 무시합니다.
        pass
    return img


def read_file_to_send(idx, total_count, file_path, resource_path, result_path, ocr_api, start_time):
    global end_flag, success_cnt, failed_cnt, json_failed_cnt, current_time, ext_cnt, count
    response = None
    json_data = config_reading()
    ocr_info = json_data['ocr_info']
    try:
        min_size = ocr_info['file_min_size']
        max_size = ocr_info['file_max_size']
        max_width_size = ocr_info['max_width_size']
        max_height_size = ocr_info['max_height_size']
        save_image_path = ocr_info["save_path"]
    except Exception as e:
        logging.error(f"OCR 설정 파일 읽기 오류 : {str(e)}")
        return
    
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    root = os.path.dirname(file_path)
    main_fold = get_main_fold(resource_path, file_path)
    file_size = os.path.getsize(file_path)

    # 파일 크기가 0인 경우 처리
    if file_size == 0:
        increment_failed_count()
        response = {
            "tags": ["ocr", "file", "N", "exception"], 
            "mime_type": "ocr_0020", 
            "exception_message": "파일 크기가 0입니다."
        }
        logging.error(f"파일 크기 오류 (0 bytes): {file_path}")
        json_failed_cnt = save_as_json(file_path, result_path, response)
        return

    try:
        # numpy로 먼저 파일을 읽어서 버퍼 생성
        with open(file_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            
        # 이미지 디코딩
        image = cv.imdecode(file_bytes, cv.IMREAD_IGNORE_ORIENTATION | cv.IMREAD_COLOR)
        
        if image is None:
            # 첫 번째 읽기 실패 시 IMREAD_ANYCOLOR로 다시 시도
            image = cv.imdecode(file_bytes, cv.IMREAD_ANYCOLOR)
            
        if image is None:
            increment_failed_count()
            response = {"tags": ["ocr", "file", "N", "exception"], "mime_type": "ocr_0019", "exception_message": "이미지 파일을 읽을 수 없습니다."}
            logging.error(f"이미지 읽기 오류: {file_path}")
            json_failed_cnt = save_as_json(file_path, result_path, response)
            return

    except Exception as e:
        increment_failed_count()
        response = {"tags": ["ocr", "file", "N", "exception"], "mime_type": "ocr_0019", "exception_message": f"이미지 파일을 읽을 수 없습니다: {str(e)}"}
        logging.error(f"이미지 읽기 오류: {file_path}, {str(e)}")
        json_failed_cnt = save_as_json(file_path, result_path, response)
        return

    if image is not None:
        try:
            height, width = image.shape[:2]  # 채널 수에 관계없이 높이와 너비만 가져오기
            
            # 그레이스케일 변환 시 예외 처리 추가
            try:
                gray_scale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            except cv.error:
                # 이미 그레이스케일인 경우
                gray_scale = image if len(image.shape) == 2 else image[:,:,0]

            if height >= min_size and width >= min_size:
                if height <= max_height_size and width <= max_width_size:
                    send_request_to_ocr(idx, image, ocr_api, resource_path, file_path, result_path)
                elif height <= 3000 and width <= 3000:
                    resize_img = cv.resize(gray_scale, None, fx=0.7, fy=0.7, interpolation=cv.INTER_LINEAR)
                    send_request_to_ocr(idx, resize_img, ocr_api, resource_path, file_path, result_path)
                elif height <= 4800 and width <= 4800:
                    resize_img = cv.resize(gray_scale, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
                    send_request_to_ocr(idx, resize_img, ocr_api, resource_path, file_path, result_path)
                else:
                    mime_type = "ocr_0001"
                    increment_ext_count()
                    indexing_ext(main_fold, file_path, result_path, mime_type, file_size)
            else:
                mime_type = "ocr_0002"
                increment_ext_count()
                indexing_ext(main_fold, file_path, result_path, mime_type, file_size)
        except Exception as e:
            increment_failed_count()
            response = {"tags": ["ocr", "file", "N", "exception"], "mime_type": "ocr_0019", 
                       "exception_message": f"이미지 처리 중 오류 발생: {str(e)}"}
            logging.error(f"이미지 처리 오류: {file_path}, {str(e)}")
            json_failed_cnt = save_as_json(file_path, result_path, response)
            return

    else:
        increment_failed_count()
        response = {"tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0005", "exception_message" : "이미지 파일이나 데이터를 읽을 수 없습니다."}
        logging.debug(f"File : {file_path}, Image is None : {response}")
        json_failed_cnt = save_as_json(file_path, result_path, response)
        write_failed_files(file_path, save_image_path, response)


    increment_progress_count()

    if count % 100 == 0:
        count_and_print_ocr_results(success_cnt, failed_cnt, total_count, count, json_failed_cnt, ext_cnt, start_time, end_flag)
    elif count % total_count == 0 or count == total_count:
        end_flag = True
        count_and_print_ocr_results(success_cnt, failed_cnt, total_count, count, json_failed_cnt, ext_cnt, start_time, end_flag)



def save_as_json(file_path, result_path, response):
    global json_failed_cnt
    reseponse = None
    json_data = config_reading()
    root_path = json_data["root_path"]
    target_path = json_data["datainfopath"]["target_path"]
    target_path = os.path.join(root_path, target_path)
    main_directory = get_main_fold(target_path, file_path)
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    now = datetime.datetime.now()
    new_uuid = uuid.uuid4()
    utc_now = datetime.datetime.utcnow()
    uuid_timestamp = utc_now.timestamp()
    uuid_str = str(new_uuid) + '_' + str(uuid_timestamp)
    new_uuid = uuid.uuid4()
    relative_path = os.path.relpath(file_path, root_path)
    root_folder = os.path.dirname(relative_path)
    full_directory = os.path.normpath(os.path.join(root_path, root_folder))
    meta_info = read_file_from_path(file_path)
    # root_folder = relative_path.split(os.sep)[0] if os.sep in relative_path else ''
    ocr_data = {}
    ocr_data["file"] = {}
    save_json_failed_path = json_data['ocr_info']['log_file_path']
    # logging.debug(full_directory)

    try:
        if response:
            ocr_data["json_write_time"] = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            ocr_data["root_path"] = main_directory
            ocr_data["directory"] = full_directory
            ocr_data["uuid"] = uuid_str
            ocr_data["file"]["accessed"] = meta_info.get("accessed", "")
            ocr_data["file"]["ctime"] = meta_info.get("created", "")
            ocr_data["file"]["mtime"] = meta_info.get("mtime", "")
            ocr_data["file"]["owner"] = meta_info.get("owner", "")
            ocr_data["file"]["path"] = file_path
            ocr_data["file"]["mime_type"] = f"image/{file_extension}"
            ocr_data["file"]["size"] = os.path.getsize(file_path)
            ocr_data["file"]["type"] = "file"
            ocr_data["title"] = f"{file_name}{file_extension}"
            ocr_data["file"]["extension"] = file_extension.lstrip('.')

            if 'words' in response:
                ocr_data["tags"] = "ocr", "file", "S"
                ocr_data["file"]["mime_type"] = f"image/{file_extension}"
                ocr_data["content"] = "".join(response.get("words", []))
                ocr_data["summary"] = "".join(response.get("words", []))[:300]
            elif 'tags' in response and 'F' in response['tags']:
                ocr_data["tags"] = response['tags']
                if "mime_type" in response:
                    ocr_data["file"]["mime_type"] = response["mime_type"]
                else:
                    ocr_data["file"]["mime_type"] = f"image/{file_extension}"
                if "content" in response:
                    ocr_data["content"] = "".join(response.get("content"))
                    ocr_data["summary"] = "".join(response.get("content"))[:300]
                else:
                    ocr_data["content"] = None
                    ocr_data["summary"] = None
                if "exception_message" in response:
                    ocr_data["exception_message"] = response["exception_message"]
            else:
                ocr_data["tags"] = response["tags"]
                if "mime_type" in response:
                    ocr_data["file"]["mime_type"] = response["mime_type"]
                if "content" in response:
                    contents = response.get("content", [])
                    ocr_data["content"] = "".join(contents)
                    ocr_data["summary"] = "".join(contents)[:300]
                if "exception_message" in response:
                    ocr_data["exception_message"] = response["exception_message"]
    except Exception as e:
        increment_json_failed_count()
        error_message = f"오류 발생: {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 구조 생성 중 알 수 없는 오류 발생"}
        logging.error(error_message)
        write_save_json_failed_files(file_path, save_json_failed_path, response)

    # json_file_name = os.path.join(result_path, f"{uuid_str}.json")
    json_extension = file_extension.lstrip('.')
    json_file_name = os.path.join(result_path, f"{file_name}_{json_extension}_{uuid_str}_{current_time}.json")

    try:
        # with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_string = json.dumps(ocr_data, ensure_ascii=False, indent=4) 
    except json.JSONDecodeError as e:
        # 유효하지 않은 JSON 구조일 경우 예외 처리
        increment_json_failed_count()
        error_message = f"JSONDecodeError : {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : "유효하지 않은 JSON 구조입니다."}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(error_message)
    except TypeError as e:
        # 변환할 수 없는 타입일 경우 예외 처리
        error_message = f"TypeError : {str(e)}"
        increment_json_failed_count()
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : "JSON으로 변환할 수 없는 구조입니다."}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(error_message)
    except Exception as e:
        increment_json_failed_count()
        error_message = f"JSON 구조 생성 중 알 수 없는 오류 발생 : {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 구조 생성 중 알 수 없는 오류 발생"}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(error_message)
    else:
        try:
            # file_write_lock = threading.Lock()
            with file_write_lock:
                with open(json_file_name, 'w', encoding='utf-8') as json_file:
                    json_file.write(json_string)
        except Exception as e:
            increment_json_failed_count()
            error_message = f"JSON 파일 생성 중 알 수 없는 오류 발생 : {str(e)}"
            response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 파일 생성 중 알 수 없는 오류 발생"}
            write_save_json_failed_files(file_path, save_json_failed_path, response)
            logging.error(error_message)


    return json_failed_cnt


def indexing_ext(main_fold, file_path, result_path, exc_file_type, file_size):
    global json_failed_cnt
    logging.debug("색인 제외 파일 JSON 생성 시작")
    save_json_failed_path = os.getcwd()
    now = datetime.datetime.now()
    json_data = config_reading()
    root_path = json_data["root_path"]
    relative_path = os.path.relpath(file_path, root_path)
    root_folder = os.path.dirname(relative_path)
    full_directory = os.path.normpath(os.path.join(root_path, root_folder))
    docmetainfo = {}
    docmetainfo["tags"] = []
    
    if file_path is None:
        return
    
    try:                 
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
        file_ext = os.path.splitext(file_path)[1].lower()
        file_ext = file_ext.lower() 
        new_uuid = uuid.uuid4()
        utc_now = datetime.datetime.utcnow()
        timestamp = utc_now.timestamp()
        uuid_str = str(new_uuid) + '_' + str(timestamp)
        # file_size = os.path.getsize(file_path) 

    except Exception as e:
        info_message = f"{file_path}, 색인 제외 JSON 생성 중 오류 발생 : {str(e)}"
        increment_json_failed_count()
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : "유효하지 않은 JSON 구조입니다."}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(info_message)
                
    docmetainfo["file"] = {}
    try:
        docmetainfo["root_path"] = main_fold 
        docmetainfo["directory"] = full_directory
        docmetainfo["title"] = f"{file_name}{file_extension}"
        docmetainfo["file"]["path"] = file_path
        docmetainfo["file"]["extension"] = file_ext[1:]
        docmetainfo["file"]["size"] = file_size
        docmetainfo["uuid"] = uuid_str 
        docmetainfo["tags"].append("file")
        docmetainfo["tags"].append("ocr")
        docmetainfo["tags"].append("N")
        docmetainfo["tags"].append("exception")
        docmetainfo["json_write_time"] = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        docmetainfo["file"]["type"] = "file"
        docmetainfo["file"]["mime_type"] = f"{exc_file_type}"
    except Exception as e:
        error_message = str(e)
        increment_json_failed_count()
        logging.error(f"색인 제외 파일 JSON 생성 중 오류 발생 : {error_message}")
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : "유효하지 않은 JSON 구조입니다."}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        return 
    
    json_extension = file_extension.lstrip('.')
    json_file_name = os.path.join(result_path, f"{file_name}_{json_extension}_{uuid_str}_{current_time}.json")

    try:
        # with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_string = json.dumps(docmetainfo, ensure_ascii=False, indent=4)
    except json.JSONDecodeError as e:
        # 유효하지 않은 JSON 구조일 경우 예외 처리
        increment_json_failed_count()
        error_message = f"JSONDecodeError : {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : "유효하지 않은 JSON 구조입니다."}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(error_message)
    except TypeError as e:
        # 변환할 수 없는 타입일 경우 예외 처리
        error_message = f"TypeError : {str(e)}"
        increment_json_failed_count()
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0009", "exception_message" : "JSON으로 변환할 수 없는 구조입니다."}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(error_message)
    except Exception as e:
        increment_json_failed_count()
        error_message = f"JSON 구조 생성 중 알 수 없는 오류 발생 : {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 구조 생성 중 알 수 없는 오류 발생"}
        write_save_json_failed_files(file_path, save_json_failed_path, response)
        logging.error(error_message)
    else:
        try:
            # file_write_lock = threading.Lock()
            with ext_file_write_lock:
                with open(json_file_name, 'w', encoding='utf-8') as json_file:
                    json_file.write(json_string)
        except Exception as e:
            increment_json_failed_count()
            error_message = f"JSON 파일 생성 중 알 수 없는 오류 발생 : {str(e)}"
            response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 파일 생성 중 알 수 없는 오류 발생"}
            write_save_json_failed_files(file_path, save_json_failed_path, response)
            logging.error(error_message)



def write_save_json_failed_files(file_path, save_json_failed_path, response):
    exception_message = response['exception_message']
    file_size = os.path.getsize(file_path)
    json_data = config_reading()
    log_file_path = json_data['ocr_info']['log_file_path']
    failed_files_path = os.path.join(log_file_path, f'json_failed_files_{current_time}.txt')

    try:
        with write_json_failed_file_write_lock:
            with open(failed_files_path, 'a', encoding='utf-8') as failed_files_file:
                failed_files_file.write(f"{file_path}, Result description : {exception_message}, File size: {file_size}\n")
    except Exception as e:
        error_message = f"{failed_files_file} 파일 만드는데에 오류 발생 : {str(e)}"
        logging.error(error_message)


        
def write_failed_files(file_path, result_path, response):
    json_data = config_reading()
    ocr_log_path = json_data['ocr_info']['log_file_path']
    failed_files_path = os.path.join(ocr_log_path, f"failed_files_list_{current_time}.txt") 

    try:
        # file_write_lock = threading.Lock()
        with write_failed_file_write_lock:
            with open(failed_files_path, 'a', encoding='utf-8') as failed_files_list:
                failed_files_list.write(f"{file_path}, {response}\n")
    except Exception as e:
        error_message = f"Failed to write to the file {failed_files_path}: {str(e)}"
        logging.debug(error_message)




def main(ocr_api, max_thread, thread_flag, start_time, result_path, resource_path):
    start_time = time.time()
    json_data = config_reading()
    onefile_flag = json_data["ocr_info"]["onefile_flag"]
    file_path = json_data["ocr_info"]["file_path"]
    futures = []
    index = 0

    image_files, total_size = find_image_files(resource_path, result_path, tuple(ocr_filter))

    total_count = len(image_files)
    logging.info(f"총 파일 개수 : {total_count}")
    logging.info(f"파일 총 용량 : {total_size:.2f}GB")


    if thread_flag:
        # 스레드로 전송
        # logging.info(f"## 스레드 전송 시작")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread) as executor:
            for img_file_path in image_files:
                main_fold = get_main_fold(resource_path, img_file_path)
                index += 1
                # logging.debug(index)
                futures.append(executor.submit(read_file_to_send, index, total_count, img_file_path, resource_path, result_path, ocr_api, start_time))
            concurrent.futures.wait(futures)
            executor.shutdown()
    # 한 파일만 전송
    elif onefile_flag:
        main_fold = get_main_fold(file_path)
        total_count = 1
        read_file_to_send(index, total_count, file_path, resource_path, result_path, ocr_api, start_time)
    else:
        # 스레드 없이 한 개씩 전송
        for img_file_path in image_files:
            read_file_to_send(index, total_count, img_file_path, resource_path, result_path, ocr_api, start_time)
        
if __name__ == "__main__":
    config_data = config_reading()
    ver = 1.0

    if config_data != None:    
        root_path = config_data['root_path']
        resource_path = config_data['datainfopath']['target_path']
        el_target_path = config_data['elasticsearch']['el_file_target_path']
        el_file_path = config_data['elasticsearch']['el_file_path']
        ocr_filter = config_data['datafilter']['image_extensions']
        ocr_info = config_data['ocr_info']
        ocr_server_ip = ocr_info['ocr_server_ip']
        ocr_server_port = ocr_info['ocr_server_port']
        max_thread = ocr_info['max_thread']               
        thread_flag = ocr_info['thread_flag']                 
        resource_path = os.path.join(root_path, resource_path)
        result_path = os.path.join(root_path, el_target_path, el_file_path)
        log_level = ocr_info['log_to_level']
        log_to_console = ocr_info['log_to_console']
        log_to_file = ocr_info['log_to_file']
        log_file_path = ocr_info['log_file_path']

        logger = logging.getLogger('')
        log_level = get_log_level(log_level)
        
        # 콘솔 로깅 설정 - config의 log_level 사용
        if log_to_console:
            console = logging.StreamHandler()
            console.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
            console.setFormatter(formatter)
            logger.addHandler(console)

        # 파일 로깅 설정 - 항상 INFO 레벨
        if log_to_file:
            log_directory = os.path.join(os.getcwd(), 'logs')
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)

            current_time = time.strftime("%Y%m%d%H%M%S")
            log_file_name = f"ocr_processor_{current_time}.log"
            log_file_path = os.path.join(log_directory, log_file_name)
            
            file_handler = RotatingFileHandler(
                filename=log_file_path,
                maxBytes=10*1024*1024,
                backupCount=10,
                encoding='utf-8'
            )
            
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - [%(levelname)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)  # 파일은 항상 INFO 레벨로 설정
            logger.addHandler(file_handler)
        
        # 로거의 기본 레벨은 가장 낮은 레벨로 설정하여 모든 핸들러가 동작하도록 함
        logger.setLevel(min(log_level, logging.INFO))

        logging.info(f"DeepLearning OCR ver : {ocr_info['ocr_ver']}")

        try:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
        except Exception as e:
            logging.debug(f"결과 폴더를 생성하는 중 오류 발생: {str(e)}")

        try:
            if not os.path.exists(resource_path):
                os.makedirs(resource_path)
        except Exception as e:
            logging.error(f"폴더를 생성하는 중 오류 발생: {str(e)}")

        if len(sys.argv) == 2:
            sys_argv = sys.argv[1]
            ocr_api = f"{sys_argv}:{ocr_server_port}"
        elif len(sys.argv) == 1:
            ocr_api = f"{ocr_server_ip}:{ocr_server_port}"

        if not os.path.exists(root_path):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.debug( f"\n Exit there is no root path  time: ", {current_time})
            sys.exit(1)
            
        # logging.debug( f"\n ocr_processor - ver : {ver}, Start Process")
        start_time = time.time()
        main(ocr_api, max_thread, thread_flag, start_time, result_path, resource_path)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # logging.debug(f"\n ocr_processor - ver: {ver}, Process End Time : {current_time}")



