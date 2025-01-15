import requests
import base64
import concurrent.futures
import time
import os
import json
import time
import datetime
import threading
import uuid
import subprocess
from collections import Counter

processed_file_cnt = 0
failed_cnt = 0
total_image_files = 0
total_image_data_size = 0
ocr_flag = False
json_flag = False
error_counts = Counter()


def config_reading():
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, 'config.json')
    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            return json_data        
    else:
        print("config.json 파일을 찾을 수 없습니다.")
        return None


def bytes_to_gb(bytes_size):
    gb_size = round(bytes_size / (1024 ** 3), 2)
    return gb_size

def extract_error_message(error_message):
    parts = error_message.split(" - ")
    return parts[-1]


def read_file_from_path(path):
    try:
        file_stats = os.stat(path)
        file_size = file_stats.st_size
        creation_time = time.ctime(file_stats.st_ctime)
        modification_time = time.ctime(file_stats.st_mtime)
        access_time = time.ctime(file_stats.st_atime)
        uid = file_stats.st_uid
        gid = file_stats.st_gid
        owner_info = f"{uid}:{gid}"

        meta_info = {
            "accessed": access_time,
            "ctime":creation_time,
            "mtime": modification_time,
            "size": file_size,
            "owner": owner_info
        }

        file_size  = file_size / 1024
        return meta_info
    except Exception as e:      
        print(f"{path}, a exceptions: {str(e)}")
        return None 


def send_request_to_api(image_data, endpoint, use_cpu=False, use_quad=False):
    b64_string = base64.b64encode(image_data).decode('utf-8')
    
    response = requests.post(endpoint, json={"image": b64_string})
   
    return response.json()



def send_request_to_ocr(path, output_path, image, api, is_group_lines=False, is_bilateral=False, use_cpu=False, use_quad=False):
    global failed_cnt, error_counts

    endpoint = f"http://{api}/ocr_all"
    filename = os.path.basename(path)
    try:
        response = send_request_to_api(image, endpoint, use_cpu=use_cpu, use_quad=use_quad)
        if response:
            print("API Response:", response)
            return response
        else:
            failed_cnt += 1
            ocr_flag = True
            error_message = f"{filename} - API 요청 실패 또는 응답 데이터 오류"
            response = {"ocr_error": error_message}
            print(error_message)
            write_failed_file(output_path, filename, response, json_flag, ocr_flag)
            error_counts[extract_error_message(error_message)] += 1 
            return None
    except Exception as e:
        ocr_flag = True
        failed_cnt += 1
        error_message = f"{filename} - OCR API 요청 중 오류 발생: {str(e)}"
        response = {"ocr_error": error_message}
        print(error_message)
        write_failed_file(output_path, filename, response, json_flag, ocr_flag)
        error_counts[extract_error_message(error_message)] += 1 
        return None


def process_image(root_path, filename, ocr_api, ocr_output_path, file_path, result_path):
    global processed_file_cnt, failed_cnt, ocr_flag

    ocr_flag = True

    try:
        response = api_test(ocr_output_path, file_path, ocr_api)
        if response is not None:
            save_as_json(root_path, result_path, file_path, response, ocr_output_path, failed_cnt, ocr_flag)
            processed_file_cnt += 1
    except Exception as e:
        failed_cnt += 1
        error_message = f"{filename} - 오류 발생: {str(e)}"
        response = {"ocr_error" : error_message}
        print(error_message)
        write_failed_file(result_path, filename, response, json_flag, ocr_flag)
        error_counts[extract_error_message(error_message)] += 1  


def process_image_multithreaded(root_path, resource_path, ocr_api, ocr_output_path, ocr_filter, ocr_thread, result_path):
    global failed_cnt, processed_file_cnt, total_image_files, total_image_data_size
    print_every = 100
    thread_id = threading.get_ident()

    resource_path = os.path.join(root_path, resource_path)
    ocr_output_path = os.path.join(resource_path, ocr_output_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=ocr_thread) as executor:
        futures = []

        for root, _, files in os.walk(resource_path):
            for filename in files:
                file_extension = os.path.splitext(filename)[-1]
                if file_extension in ocr_filter:
                    total_image_files += 1
                    file_path = os.path.join(root, filename)
                    total_image_data_size += os.path.getsize(file_path)
                    future = executor.submit(process_image, root_path, filename, ocr_api, ocr_output_path, file_path, result_path)
                    futures.append(future)

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            if idx % print_every == 0:
                print(f"[INFO] ocr processor {thread_id} 진행 상황")
                print(f"총 이미지 파일 개수 : {processed_file_cnt + failed_cnt}개")
                print(f"성공 파일 개수 : {processed_file_cnt}개")
                print(f"분석 실패한 파일 개수: {failed_cnt}개")

    total_image_data_gb = bytes_to_gb(total_image_data_size)
    total_image_files = processed_file_cnt + failed_cnt

    return total_image_files, failed_cnt, total_image_data_gb


def api_test(output_path, path, api):
    with open(path, 'rb') as image_file:
        image_data = image_file.read()
    response = send_request_to_ocr(path, output_path, image_data, api, is_group_lines=False, is_bilateral=False, use_cpu=False, use_quad=False)

    return response


def save_as_json(root_path, result_path, path, response, output_path, failed_cnt, ocr_flag):
    meta_info = read_file_from_path(path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    filename = os.path.basename(path)
    json_filename = os.path.join(output_path, f"{filename}_{timestamp}.json")
    extension = os.path.splitext(filename)[-1]
    new_uuid = uuid.uuid4()
    uuid_str = str(new_uuid) + '_' + str(timestamp)
    ocr_data = {}
    ocr_data["file"] = {}
    path = path.replace("\\", "/")
    directory = os.path.dirname(path)


    ocr_data["@timestamp"] = timestamp
    ocr_data["title"] = filename
    ocr_data["content"] = "".join(response.get("words", []))
    ocr_data["summary"] = response.get("words", "")[:300]
    ocr_data["root_path"] = root_path
    ocr_data["directory"] = directory
    ocr_data["type"] = "json"
    ocr_data["uuid"] = uuid_str
    ocr_data["file"]["accessed"] = meta_info["accessed"]                                                                                                                                                                                                                                       
    ocr_data["file"]["ctime"] = meta_info["ctime"]
    ocr_data["file"]["mtime"] = meta_info["mtime"]
    ocr_data["file"]["owner"] = meta_info["owner"]
    ocr_data["file"]["path"] = path
    ocr_data["file"]["extension"] = extension
    ocr_data["file"]["mime_type"] = f"image/{extension}"
    ocr_data["file"]["size"] = meta_info["size"]
    ocr_data["file"]["type"] = f"image/{extension}"

    try:
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(ocr_data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        json_flag = True
        failed_cnt += 1
        error_message = f"JSON 파일 저장 중 오류 발생: {str(e)}"
        response = {"json_error": error_message}
        write_failed_file(result_path, filename, response, ocr_flag, json_flag)
        print(error_message)


def write_failed_file(result_path, filename, response, json_flag, ocr_flag):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if ocr_flag:
        error_message = response['ocr_error']
        failed_files_path = os.path.join(result_path, 'ocr_failed_files.txt')
    elif json_flag:
        error_message = response['json_error']
        failed_files_path = os.path.join(result_path, 'json_failed_files.txt')

    with open(failed_files_path, 'a', encoding='utf-8') as failed_files_file:
        failed_files_file.write(f"파일명: {filename}, {error_message}\n")



def main():
    json_data = config_reading()
    root_path = json_data['root_path']
    resource_path = json_data['resource_path']
    result_path = json_data['result_path']
    ocr = json_data['ocr']
    ocr_thread = ocr['thread']
    ocr_api = ocr['nts_api']
    ocr_filter = ocr['filter']
    ocr_output_path = ocr['output_path']

    start_time = time.time()
    total_img_cnt, tika_failed_cnt, total_image_data_gb = process_image_multithreaded(root_path, resource_path, ocr_api, ocr_output_path, ocr_filter, ocr_thread, result_path)
    end_time = time.time()
    total_time = end_time - start_time
    processed_img_cnt = total_img_cnt - tika_failed_cnt 

    print("=====[INFO] ocr-processing 결과=====")
    print(f"총 이미지 파일 개수 : {total_img_cnt}개")
    print(f"총 이미지 파일 크기 : {total_image_data_gb}GB")
    print(f"OCR 처리 완료 파일 개수 : {processed_img_cnt}개")
    print(f"OCR 처리 실패 파일 개수 : {tika_failed_cnt}개")
    print(f"파일 총 처리 시간 : {total_time:.2f}초")
    
    for error_message, count in error_counts.items():
        print(f"{error_message}, 발생 횟수 : {count}")


if __name__ == "__main__":
    main()

