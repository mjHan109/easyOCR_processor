import requests
import os
import logging
import threading
import time
import json
import datetime
import uuid
import io
import sys
import magic
import struct
import aiohttp
from PIL import Image, PngImagePlugin
import asyncio
from langdetect import detect
import pytesseract
from wand.image import Image as WandImage
import tifffile
import subprocess

# 전역 변수
success_count = 0
failure_count = 0

def setup_logging(log_to_console, log_to_file, log_file_path, log_level):
    """
    로깅 설정
    Parameters:
        log_to_console (bool): 콘솔 로깅 여부
        log_to_file (bool): 파일 로깅 여부
        log_file_path (str): 로그 파일 경로
        log_level (str): 로그 레벨
    """
    logger = logging.getLogger('')
    logger.setLevel(log_level)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logging.info("Logging setup complete.")

def config_reading(config_file):
    """
    JSON 설정 파일 읽기
    Parameters:
        config_file (str): 설정 파일 경로
    Returns:
        dict: JSON 설정 데이터
    """
    import json
    if not os.path.exists(config_file):
        # logging.error(f"Config file {config_file} not found.")
        return None

    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
            # logging.info("Configuration loaded successfully.")
            return config_data
    except Exception as e:
        # logging.error(f"Error reading config file {config_file}: {str(e)}")
        return None

def remove_text_chunks(file_path):
    """
    PNG 파일에서 텍스트 청크 제거
    Parameters:
        file_path (str): 파일 경로
    Returns:
        bytes: 텍스트 청크 제거된 PNG 데이터
    """
    try:
        with Image.open(file_path) as img:
            if img.format != 'PNG':
                logging.warning(f"Not a PNG file: {file_path}")
                return None
            data = io.BytesIO()
            img.save(data, format='PNG', pnginfo=None)
            data.seek(0)
            # logging.info(f"Removed text chunks from: {file_path}")
            return data.getvalue()
    except Exception as e:
        # logging.error(f"Failed to remove text chunks from {file_path}: {str(e)}")
        return None

def is_valid_image(file_path):
    """
    파일 유효성 검증
    Parameters:
        file_path (str): 파일 경로
    Returns:
        bool: 유효한 이미지 여부
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.error(f"Invalid image file {file_path}: {str(e)}")
        return False

def process_image(file_path, json_data):
    """
    단일 이미지 처리
    Parameters:
        file_path (str): 파일 경로
        json_data (dict): JSON 설정 데이터
    """
    try:
        # macOS 메타데이터 파일 필터링
        if file_path.startswith('._') or '__MACOSX' in file_path:
            logging.info(f"Skipping macOS metadata file: {file_path}")
            return

        # PNG 텍스트 청크 제거
        if file_path.lower().endswith('.png'):
            image_data = remove_text_chunks(file_path)
            if not image_data:
                logging.error(f"Skipping invalid PNG file: {file_path}")
                return
        else:
            # 일반 파일 읽기
            with open(file_path, 'rb') as f:
                image_data = f.read()

        # 유효성 검증
        if not is_valid_image(file_path):
            # logging.error(f"Skipping invalid image file: {file_path}")
            return

        # 처리 성공 카운트 증가
        global success_count
        success_count += 1
        # logging.info(f"Successfully processed image: {file_path}")

    except Exception as e:
        global failure_count
        failure_count += 1
        # logging.error(f"Error processing image {file_path}: {str(e)}")

def process_images(image_files, json_data):
    """
    이미지 파일 리스트 처리
    Parameters:
        image_files (list): 이미지 파일 리스트
        json_data (dict): JSON 설정 데이터
    """
    for file_path in image_files:
        process_image(file_path, json_data)

def filter_large_images(image_files, json_data):
    """
    대용량 이미지를 필터링하여 별도 리스트로 분리
    Parameters:
        image_files (list): 이미지 파일 리스트
        json_data (dict): JSON 설정 데이터
    Returns:
        tuple: (필터링된 이미지 리스트, 대용량 이미지 리스트)
    """
    filtered_files = []
    big_size_files = []
    max_pixel_limit = json_data['ocr_info']['max_pixel_limit']

    for file_path in image_files:
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if width * height > max_pixel_limit:
                    logging.warning(f"Image too large: {file_path}")
                    big_size_files.append(file_path)
                else:
                    filtered_files.append(file_path)
        except Exception as e:
            logging.error(f"Error filtering image {file_path}: {str(e)}")

    return filtered_files, big_size_files

def save_as_json(file_path, uuid_str, response, json_data, success):
    global json_failed_count, failure_count, success_count
    """
    JSON 파일로 저장하는 함수

    Parameters:
        file_path (str): 이미지 파일 경로
        uuid_str (str): UUID 문자열
        response (dict): OCR 서버 응답 데이터
        json_data (dict): 설정 데이터
        success (bool): 분석 성공 여부
    """

    root_path = json_data["root_path"]
    target_path = json_data["datainfopath"]["target_path"]
    target_path = os.path.join(root_path, target_path)
    main_directory = get_main_fold(target_path, file_path)
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    save_json_failed_path = os.getcwd()
    es_target_path = json_data['elasticsearch']['normal_el_file_target_path']
    es_filepath = json_data['elasticsearch']['el_file_path']
    now = datetime.datetime.now()
    # new_uuid = uuid.uuid4()
    # utc_now = datetime.datetime.utcnow()
    # timestamp = utc_now.timestamp()
    # uuid_str = str(new_uuid) + '_' + str(timestamp)
    
    relative_path = os.path.relpath(file_path, root_path)
    root_folder = os.path.dirname(relative_path)
    full_directory = os.path.normpath(os.path.join(root_path, root_folder))
    meta_info = read_file_from_path(file_path)
    
    if not os.path.exists(es_target_path):
        os.mkdir(es_target_path)

    if not os.path.exists(es_filepath):
        os.mkdir(es_filepath)

    result_path = os.path.join(es_target_path, es_filepath)

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    

    ocr_data = {}
    ocr_data["file"] = {}
    
    try:
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

        if success and response is not None:
            if isinstance(response, str):
                try:
                    response = eval(response)
                except:
                    response = {"text": response}
            
            ocr_data["tags"] = ["ocr", "file", "S"]
            ocr_data["content"] = response.get("text", "")            
            # summary는 content의 처음 300자로 설정
            content_text = response.get("text", "")
            ocr_data["summary"] = content_text[:300]
            success_count += 1
        else:
            failure_count += 1
            ocr_data["tags"] = ["ocr", "file", "N", "exception"]
    
    except Exception as e:
        error_message = f"오류 발생: {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"]}

    json_file_name = os.path.join(result_path, f"{uuid_str}.json")

    try:
        # with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_string = json.dumps(ocr_data, ensure_ascii=False, indent=4) 
    except Exception as e:
        failure_count += 1
        error_message = f"JSON 구조 생성 중 알 수 없는 오류 발생 : {str(e)}"
        logging.error(error_message)
    else:
        try:
            with open(json_file_name, 'w', encoding='utf-8') as json_file:
                json_file.write(json_string)
        except Exception as e:
            json_failed_count +=1
            error_message = f"JSON 파일 생성 중 알 수 없는 오류 발생 : {str(e)}"
            logging.error(error_message)

    return json_failed_count


def main():
    """
    메인 함수
    """
    # JSON 설정 파일 읽기
    json_data = config_reading('config.json')
    if not json_data:
        return

    try:
        # 설정 초기화
        source_path = os.path.join(json_data['root_path'], json_data['datainfopath']['source_path'])
        image_extensions = json_data['datafilter']['image_extensions']
        batch_size = json_data['ocr_info']['batch_size']
        
        # 로깅 설정
        setup_logging(
            json_data['ocr_info']['log_to_console'],
            json_data['ocr_info']['log_to_file'],
            f"ocr_processor_{time.strftime('%Y%m%d_%H%M%S')}.log",
            json_data['ocr_info']['log_to_level']
        )

        # 이미지 파일 수집
        image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(source_path)
            for file in files
            if any(file.lower().endswith(ext) for ext in image_extensions)
        ]

        # 대용량 이미지 필터링
        filtered_files, big_size_files = filter_large_images(image_files, json_data)

        # 배치 처리
        start_time = time.time()
        for i in range(0, len(filtered_files), batch_size):
            batch = filtered_files[i:i + batch_size]
            process_images(batch, json_data)

        # 결과 출력
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"총 이미지 파일 개수: {len(image_files)}")
        logging.info(f"총 분석 시간: {hours}시간, {minutes}분, {seconds:.2f}초")
        logging.info(f"분석 성공 파일: {success_count}개, 분석 실패 파일: {failure_count}")

    except Exception as e:
        logging.error(f"Main process error: {str(e)}")

if __name__ == "__main__":
    main()
