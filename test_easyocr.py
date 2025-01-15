import easyocr
import os
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
import torch
import datetime
import threading


json_failed_cnt = 0

write_json_failed_file_write_lock = threading.Lock()
json_failed_lock = threading.Lock()
file_write_lock = threading.Lock()

current_time = time.strftime("%Y%m%d%H%M%S")

def increment_json_failed_count():
    global json_failed_cnt
    with json_failed_lock:
        json_failed_cnt += 1
  

def config_reading(json_file_name):
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, json_file_name)

    print(f"{current_directory}, {config_file}")
    
    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        logging.error(f"{current_directory} - {json_file_name} 파일을 찾을 수 없습니다.")
        return None


#def perform_ocr(image_path, languages=['en', 'ko']):
def perform_ocr(image_path, languages, use_gpu):
    """이미지 경로를 받아 OCR을 수행하고 결과를 반환하는 함수"""
    try:
        # EasyOCR Reader 객체 생성
        reader = easyocr.Reader(languages, gpu=use_gpu)  # 지정된 언어 지원

        # 이미지에 대해 OCR 수행
        ocr_results = reader.readtext(image_path)

        # 텍스트만 추출
        extracted_texts = [item[1] for item in ocr_results]

        # 추출된 텍스트를 하나의 문자열로 결합
        result_string = ''.join(extracted_texts)

        return result_string

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def save_to_json(data, output_file_path):
    """데이터를 JSON 파일로 저장하는 함수"""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Text results have been written to {output_file_path}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")



def get_main_fold(target_path, file_path):
    relative_path = os.path.relpath(file_path, target_path)
    components = os.path.normpath(relative_path).split(os.path.sep)
    main_fold = components[0] if len(components) > 1 else None
    
    if main_fold:
        parts = main_fold.split('_')
        filtered_parts = [part for part in parts if not part.isdigit()]
        main_fold = '_'.join(filtered_parts)
    
    return main_fold

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
        print(f"{info_message}")
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

def write_save_json_failed_files(file_path, save_json_failed_path, response):
    exception_message = response['exception_message']
    file_size = os.path.getsize(file_path)
    failed_files_path = os.path.join(save_json_failed_path, f'json_failed_files_{current_time}.txt')

    try:
        # file_write_lock = threading.Lock()
        # with write_json_failed_file_write_lock:
            with open(failed_files_path, 'a', encoding='utf-8') as failed_files_file:
                failed_files_file.write(f"{file_path}, Result description : {exception_message}, File size: {file_size}\n")
    except Exception as e:
        error_message = f"{failed_files_file} 파일 만드는데에 오류 발생 : {str(e)}"
        logging.error(error_message)

json_failed_cnt=0

def save_as_json(file_path, result_path, response, json_data):
    global json_failed_cnt
    reseponse = None

    root_path = json_data["root_path"]
    target_path = json_data["datainfopath"]["target_path"]
    target_path = os.path.join(root_path, target_path)
    main_directory = get_main_fold(target_path, file_path)
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))

    now = datetime.datetime.now()
    new_uuid = uuid.uuid4()

    utc_now = datetime.datetime.utcnow()
    timestamp = utc_now.timestamp()
    uuid_str = str(new_uuid) + '_' + str(timestamp)
    
    relative_path = os.path.relpath(file_path, root_path)
    root_folder = os.path.dirname(relative_path)
    full_directory = os.path.normpath(os.path.join(root_path, root_folder))
    meta_info = read_file_from_path(file_path)
    
    ocr_data = {}
    ocr_data["file"] = {}
    save_json_failed_path = json_data['ocr_info']['log_file_path']
    
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

            ocr_data["tags"] = "ocr", "file", "S"
            ocr_data["file"]["mime_type"] = f"image/{file_extension}"
            ocr_data["content"] = f"""{response}"""
            ocr_data["summary"] = f"""{response[:300]}"""

        else:
            ocr_data["tags"] = response["tags"]
            if "mime_type" in response:
                ocr_data["file"]["mime_type"] = response["mime_type"]
            if "content" in response:
                contents = response.get("content", [])
                ocr_data["content"] = "".join(contents)
                ocr_data["summary"] = "".join(contents)[:300]
            else:
                ocr_data["content"] = None
                ocr_data["summary"] = None
            if "exception_message" in response:
                ocr_data["exception_message"] = response["exception_message"]
    
        #print(ocr_data)

    except Exception as e:
        error_message = f"오류 발생: {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 구조 생성 중 알 수 없는 오류 발생"}
        logging.error(error_message)
        write_save_json_failed_files(file_path, save_json_failed_path, response)

    # json_file_name = os.path.join(result_path, f"{uuid_str}.json")
    json_extension = file_extension.lstrip('.')
    json_file_name = os.path.join(result_path, f"{file_name}_{json_extension}_{uuid_str}.json")

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
            # with file_write_lock:
                with open(json_file_name, 'w', encoding='utf-8') as json_file:
                    json_file.write(json_string)
        except Exception as e:
            increment_json_failed_count()
            error_message = f"JSON 파일 생성 중 알 수 없는 오류 발생 : {str(e)}"
            response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"], "mime_type" : "ocr_0012", "exception_message" : "JSON 파일 생성 중 알 수 없는 오류 발생"}
            write_save_json_failed_files(file_path, save_json_failed_path, response)
            logging.error(error_message)


    return json_failed_cnt





def main():
    """메인 함수"""

    try:
        json_data = config_reading('config.json')

        if json_data is not None:
            datafilter = json_data['datafilter']
            image_extensions = datafilter['image_extensions']
            max_cpu_use = json_data['cpu_use_persent']
            max_mem_use = json_data['memory_use_persent']
            tika_app = json_data['tika_app']
            tika_server_ip = tika_app['tika_server_ip']
            tika_server_port = tika_app['tika_server_port']
            tika_ocr_server_count = tika_app['tika_ocr_server_count']
            tika_ocr_process_num = tika_app['tika_ocr_process_num']

            ocr_info = json_data['ocr_info']
            ocr_languages = ocr_info['ocr_languages']

            root_path = json_data['root_path']
            source_path = json_data['datainfopath']['target_path']
            source_path = os.path.join(root_path, source_path)

            el_target_path = json_data['elasticsearch']['el_file_target_path']
            el_file_path = json_data['elasticsearch']['el_file_path']
            result_path = os.path.join(root_path, el_target_path, el_file_path)

            log_level = ocr_info['log_to_level']
            log_file = ocr_info['log_file']
            log_to_console = ocr_info['log_to_console']

            log_file_path = ocr_info['log_file_path']
            ocr_ver = ocr_info["ocr_ver"]

            if not os.path.exists(log_file_path):
                os.mkdir(log_file_path)

            if not os.path.exists(result_path):
                os.mkdir(result_path)

            if not os.path.exists(log_file_path):
                os.mkdir(log_file_path)

            current_time = time.strftime("%Y%m%d%H%M%S")
            log_file_name = log_file + "_" + current_time + ".log"
            log_file_path = os.path.join(log_file_path, log_file_name)

            logger = logging.getLogger('')
            log_level = get_log_level(log_level)
            logger.setLevel(log_level)

            file_handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024*1024, backupCount=7, encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            logging.getLogger("PIL").setLevel(logging.WARNING)
        
            if log_to_console:
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                console.setFormatter(formatter)
                logging.getLogger('').addHandler(console)

            logging.info(f"ocr_processor ver {ocr_ver}")

    except Exception as e:
        error_log = f"config.json 을 읽는 도중 오류 발생 : {str(e)}"
        logging.error(f"{error_log}")        
        return
    
    info_message = f"ocr_languages: {ocr_languages}"
    logging.info(info_message)

    # GPU 사용 가능 여부 확인
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        info_message = f"GPU is available. Using GPU for OCR."
        logging.info(info_message)
    else:
        info_message = f"GPU is not available. Using CPU for OCR."
        logging.info(info_message)

    # image list init.
    image_files = []
    for root, _, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)

    info_message = f"image_files list count: {len(image_files)}"
    logging.info(info_message)

    
    # OCR 수행
    start_time = time.time()
    
    combined_results = []

    count = 0
# case 1 : 하나씩 하는게 시간 및 리소스 사용량이 적은듯.
    for image_file in image_files:
        result = perform_ocr(image_file, ocr_languages, use_gpu)
        count += 1

        info_message = f"\ncount: {count}\n{image_file}\n{result}"
        logging.info(info_message)

        if result != None:
            combined_results.append(result)
            save_as_json(image_file, result_path, result, json_data)

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    info_message = f"combined_results count: {len(combined_results)}"
    logging.info(info_message)
    
    info_message = f"Total_time : {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"
    logging.info(info_message)

if __name__ == "__main__":
    main()