import asyncio
from logging.handlers import RotatingFileHandler
from multiprocessing import Pool, cpu_count
import aiohttp
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
from PIL import Image
from langdetect import detect
import pytesseract
import gi
gi.require_version('Rsvg', '2.0')
from gi.repository import Rsvg
import cairo
import io
import tifffile
import subprocess
from logging.handlers import RotatingFileHandler
import warnings


success_count = 0
failure_count = 0
json_failed_count = 0
current_time = time.strftime("%Y%m%d%H%M%S")
ocr_ver = 1.0

# PIL 관련 경고 필터 설정
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')


def get_log_level(log_level):
    """
    로그 레벨을 가져오는 함수
    
    Parameters:
        log_level (str): 로그 레벨
    
    Returns:
        int: 로그 레벨
    """
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

def setup_logging(log_to_console, log_to_file, log_file_path, log_level):
    """
    로그 설정 함수
    
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
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    if log_to_file:
        file_handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024*1024, backupCount=7, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # PIL 관련 로그 레벨 설정
    logging.getLogger('PIL').setLevel(logging.ERROR)

def config_reading(json_file_name):
    """
    설정 파일을 읽는 함수
    
    Parameters:
        json_file_name (str): 설정 파일 이름
    
    Returns:
        dict: 설정 데이터 또는 None (파일을 찾을 수 없는 경우)
    """
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, json_file_name)

    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        logging.error(f"{current_directory} - {json_file_name} 파일을 찾을 수 없습니다.")
        return None
    
import re

def get_safe_path(path):
    """
    특수 문자를 경로에서 안전하게 처리하는 함수.
    공백은 유지하며, 특수 문자만 이스케이프 처리.
    
    Parameters:
        path (str): 경로 문자열
        
    Returns:
        str: 안전하게 처리된 경로
    """
    # 이스케이프가 필요한 특수 문자 리스트 (공백은 제외)
    special_chars = r'[\[\]\(\)\{\}\*\?\!\#\$\&\+\|\^\~]'
    
    # 특수 문자를 이스케이프 처리
    safe_path = re.sub(special_chars, lambda match: f'\\{match.group()}', path)
    
    # 경로 구분자를 일관되게 슬래시(`/`)로 변환
    safe_path = os.path.normpath(safe_path).replace('\\', '/')
    
    return safe_path


def get_main_fold(target_path, file_path):
    """
    target_path에서 file_path의 경로를 제외한 상위 폴더 경로를 반환하는 함수
    
    Parameters:
        target_path (str): 대상 경로
        file_path (str): 파일 경로
        
    Returns:
        str: 상위 폴더 경로
    """
    relative_path = os.path.relpath(file_path, target_path)
    parent_dir = os.path.dirname(relative_path)
    main_directory = os.path.join(target_path, parent_dir)
    return main_directory

def initialize_uuid_map(image_files):
    """
    이미지 파일 경로와 UUID를 매핑하는 함수

    Parameters:
        image_files (list): 이미지 파일 경로 목록

    Returns:
        dict: file_path와 uuid 매핑 정보
    """
    uuid_map = {}
    for file_path in image_files:
        new_uuid = uuid.uuid4()
        utc_now = datetime.datetime.utcnow()
        timestamp = utc_now.timestamp()
        uuid_map[file_path] = f"{new_uuid}_{timestamp}"
    return uuid_map


def read_file_from_path(file_path):
    """
    파일 경로에서 파일 정보를 읽는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        dict: 파일 정보
    """
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
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)

        meta_info = {
            "accessed": access_time,
            "created":creation_time,
            "mtime": modification_time,
            "size": file_size,
            "owner": owner_info,
            "mime_type" : mime_type
        }

        return meta_info
    except Exception as e:      
        info_message = f"{file_path}, a exceptions: {str(e)}"
        print(f"{info_message}")
        return None

def get_jpeg_size(file_path):
    """
    JPEG 파일의 해상도(가로, 세로 픽셀)를 헤더에서 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(0)
            if f.read(2) != b'\xFF\xD8':  # JPEG 시그니처 확인
                raise ValueError("올바른 JPEG 파일이 아닙니다")
                
            while True:
                byte = f.read(1)
                if not byte:  # EOF 체크
                    raise ValueError("올바른 JPEG 마커를 찾을 수 없습니다")
                    
                while byte and byte != b'\xFF':
                    byte = f.read(1)
                    
                if not byte:  # EOF 체크
                    raise ValueError("올바른 JPEG 마커를 찾을 수 없습니다")
                    
                marker = f.read(1)
                if not marker:  # EOF 체크
                    raise ValueError("올바른 JPEG 마커를 찾을 수 없습니다")
                    
                if marker in [b'\xC0', b'\xC2']:  # SOF0 or SOF2 (Start of Frame markers)
                    f.read(3)  # 길이와 정밀도 건너뛰기
                    height, width = struct.unpack(">HH", f.read(4))
                    return width, height
                else:
                    size_bytes = f.read(2)
                    if not size_bytes or len(size_bytes) != 2:
                        raise ValueError("올바른 JPEG 세그먼트 크기를 읽을 수 없습니다")
                    size = struct.unpack(">H", size_bytes)[0]
                    f.read(size - 2)  # 다음 마커로 이동
                    
    except Exception as e:
        # logging.error(f"JPEG 크기 읽기 실패: {file_path}, {str(e)}")
        return None, None

def get_png_size(file_path):
    """
    PNG 파일의 해상도(가로, 세로 픽셀)를 헤더에서 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    with open(file_path, 'rb') as f:
        f.read(8)  # PNG 파일의 시그니처 건너뛰기
        chunk_header = f.read(8)  # IHDR (Image Header) 부분
        width, height = struct.unpack(">II", f.read(8))
        return width, height

def get_gif_size(file_path):
    """
    GIF 파일의 해상도(가로, 세로 픽셀)를 헤더에서 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    with open(file_path, 'rb') as f:
        f.seek(6)  # GIF 시그니처 건너뛰기 (6바이트)
        width, height = struct.unpack("<HH", f.read(4))  # 가로, 세로는 4바이트
        return width, height

def get_bmp_size(file_path):
    """
    BMP 파일의 해상도(가로, 세로 픽셀)를 헤더에서 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    with open(file_path, 'rb') as f:
        f.seek(18)  # BMP 헤더의 18번째 바이트에 가로, 세로 정보가 있음
        width, height = struct.unpack("<II", f.read(8))
        return width, height

def get_tiff_size(file_path):
    """
    TIFF 파일에서 해상도 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    with open(file_path, 'rb') as f:
        f.seek(4)  # Endian 체크 (II는 리틀 엔디안, MM은 빅 엔디안)
        endian = f.read(2)
        if endian == b'II':  # 리틀 엔디안
            fmt = '<'
        elif endian == b'MM':  # 빅 엔디안
            fmt = '>'
        else:
            raise ValueError("Invalid TIFF file format")
        # IFD 위치 확인
        f.seek(4)
        offset = struct.unpack(fmt + 'L', f.read(4))[0]
        f.seek(offset)

        # IFD에서 이미지 크기 찾기
        num_entries = struct.unpack(fmt + 'H', f.read(2))[0]
        for i in range(num_entries):
            tag = struct.unpack(fmt + 'H', f.read(2))[0]
            f.read(10)
            if tag == 256:  # ImageWidth
                width = struct.unpack(fmt + 'L', f.read(4))[0]
            elif tag == 257:  # ImageLength
                height = struct.unpack(fmt + 'L', f.read(4))[0]
        return width, height

def get_tiff_size_tifffile(file_path, uuid_map, exception_list):
    """
    Tifffile 라이브러리를 사용하여 TIFF 파일의 해상도를 가져오는 함수

    Parameters:
        file_path (str): TIFF 파일 경로

    Returns:
        tuple: 가로(width), 세로(height) 픽셀 크기
    """
    try:
        # logging.info(f"TIF/TIFF GET 사이즈 시작: {file_path}")
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            width, height = page.imagewidth, page.imagelength
            return width, height
    except Exception as e:
        handle_failure(file_path, uuid_map, exception_list)
        # logging.error(f"Error reading TIFF file {file_path} with tifffile: {str(e)}")
        return None, None

def convert_tiff_to_png(file_path):
    """
    시스템 명령을 사용하여 TIFF 파일을 PNG로 변환하여 호환성을 확보하는 함수

    Parameters:
        file_path (str): TIFF 파일 경로

    Returns:
        bytes: 변환된 PNG 이미지 데이터
    """
    try:
        # ImageMagick을 사용하여 무압축 PNG로 변환
        png_path = file_path.replace('.tif', '.png')
        subprocess.run(['convert', file_path, png_path], check=True)
        
        # 변환된 PNG 파일을 불러와 메모리에 저장
        with open(png_path, 'rb') as png_file:
            return png_file.read()
    except subprocess.CalledProcessError as e:
        logging.error(f"ImageMagick conversion failed for {file_path}: {str(e)}")
        return None

def get_image_size_from_metadata(file_path, uuid_map, exception_list):
    """
    파일 크기를 읽고 실패 시 처리하는 함수
    
    Parameters:
        file_path (str): 파일 경로
        uuid_map (dict): file_path와 uuid 매핑 정보
        exception_list (list): 실패한 파일 목록

    Returns:
        tuple: 가로(width), 세로(height) 크기
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in ['.jpg', '.jpeg']:
            return get_jpeg_size(file_path)
        elif ext == '.png':
            return get_png_size(file_path)
        elif ext == '.gif':
            return get_gif_size(file_path)
        elif ext == '.bmp':
            return get_bmp_size(file_path)
        elif ext in ['.tiff', '.tif']:
            return get_tiff_size_tifffile(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        handle_failure(file_path, uuid_map, exception_list)
        return None, None

def convert_tiff_to_png(file_path):
    """
    시스템 명령을 사용하여 TIFF 파일을 PNG로 변환하여 호환성을 확보하는 함수

    Parameters:
        file_path (str): TIFF 파일 경로

    Returns:
        bytes: 변환된 PNG 이미지 데이터
    """
    try:
        # ImageMagick을 사용하여 무압축 PNG로 변환
        png_path = file_path.replace('.tif', '.png')
        subprocess.run(['convert', file_path, png_path], check=True)
        
        # 변환된 PNG 파일을 불러와 메모리에 저장
        with open(png_path, 'rb') as png_file:
            return png_file.read()
    except subprocess.CalledProcessError as e:
        logging.error(f"ImageMagick conversion failed for {file_path}: {str(e)}")
        return None
    
def is_file_accessible(file_path):
    """
    파일 접근성 검증 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        bool: 파일 접근 가능 여부
    """
    try:
        with open(file_path, 'rb') as f:
            f.read(10)  # 파일의 앞부분만 읽기

        # PIL을 이용하여 이미지 검증 (이 부분을 추가)
        with Image.open(file_path) as img:
            img.verify()  # 이미지가 손상되지 않았는지 확인
        return True

    except (IOError, SyntaxError) as e:
        # logging.info(f"File {file_path} is not accessible or is corrupted: {str(e)}")
        return False

def process_image(img, file_path):
    """
    이미지를 처리하고 적절한 형식으로 변환하는 함수
    
    Parameters:
        file_path (str): 이미지 파일 경로
        
    Returns:
        bytes: 처리된 이미지 데이터
    """
    try:
        with Image.open(file_path) as img:
            # 팔레트 이미지 처리
            if img.mode == 'P':
                # 투명도가 있는 경우
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            
            # RGBA 이미지 처리
            if img.mode == 'RGBA':
                # 흰색 배경에 이미지 합성
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.split()[3]:  # 알파 채널이 있는 경우
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                img = background
            
            # 기타 모드 처리
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            # 이미지를 바이트로 변환
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            return img_byte_arr.getvalue()
            
    except Exception as e:
        logging.error(f"이미지 처리 중 오류 발생 - {file_path}: {str(e)}")
        return None

def handle_failure(file_path, uuid_map, exception_list):
    """
    실패한 파일을 처리하는 함수

    Parameters:
        file_path (str): 파일 경로
        uuid_map (dict): file_path와 uuid 매핑 정보
        exception_list (list): 실패한 파일 목록
    """
    if file_path not in exception_list:
        exception_list.append(file_path)
    # logging.error(f"Failed to process file: {file_path}, UUID: {uuid_map.get(file_path, 'unknown')}")

def process_exceptions(exception_list, uuid_map, json_data):
    """
    실패한 파일을 처리하는 함수

    Parameters:
        exception_list (list): 실패한 파일 목록
        uuid_map (dict): file_path와 uuid 매핑 정보
    """
    for file_path in exception_list:
        uuid_str = uuid_map.get(file_path)
        save_as_json(file_path, uuid_str, {}, json_data, success=False)
        # logging.info(f"Reprocessing failed file: {file_path}, UUID: {uuid_str}")

def detect_language(image_data):
    try:
        if isinstance(image_data, str):
            image_data = image_data.encode()

        if isinstance(image_data, io.BytesIO):
            image_data = image_data.getvalue()
            
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image, lang='kor+eng')

        if not text or len(text.strip()) < 5:  # 최소 길이 조정
            return "ko"  # 기본값으로 한국어 설정
            
        try:
            language = detect(text)
            return language
        except:
            return "ko"  # 감지 실패시 한국어로 설정
            
    except Exception as e:
        # logging.error(f"언어 감지 실패: {str(e)}")
        return "ko"  # 예외 발생시 한국어로 설정

# def resize_image_in_memory(file_path, json_data, uuid_map, exception_list):
#     """
#     이미지를 리사이즈하고 실패 시 처리하는 함수

#     Parameters:
#         file_path (str): 이미지 파일 경로
#         json_data (dict): 설정 데이터
#         uuid_map (dict): file_path와 uuid 매핑 정보
#         exception_list (list): 실패한 파일 목록

#     Returns:
#         bytes: 리사이즈된 이미지 데이터
#     """
#     try:
#         max_width = json_data['ocr_info']['max_width']
#         max_height = json_data['ocr_info']['max_height']
#         max_pixel_limit = json_data['ocr_info']['max_pixel_limit']

#         with Image.open(file_path) as img:
#             width, height = img.size
#             if width * height > max_pixel_limit:
#                 scale = min(max_width / width, max_height / height)
#                 img = img.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
            
#             img_byte_arr = io.BytesIO()
#             img.save(img_byte_arr, format='PNG')
#             return img_byte_arr.getvalue()
#     except Exception as e:
#         handle_failure(file_path, uuid_map, exception_list)
#         return None

# def resize_image_in_memory(file_path, img, width, height, json_data, exception_list):
#     """
#     이미지를 메모리에서 리사이즈하는 함수

#     Parameters:
#         img (PIL.Image): PIL Image 객체
#         width (int): 현재 이미지 너비
#         height (int): 현재 이미지 높이
#         json_data (dict): 설정 데이터
#         exception_list (list): 실패한 파일 목록
#     Returns:
#         bytes: 리사이즈된 이미지 데이터
#     """
#     try:
#         max_width = json_data['ocr_info']['max_width']
#         max_height = json_data['ocr_info']['max_height']
        
#         # 리사이즈가 필요한지 확인
#         if width <= max_width and height <= max_height:
#             if img.mode != 'L':
#                 img = img.convert('L')
#             img_byte_arr = io.BytesIO()
#             img.save(img_byte_arr, format='PNG')
#             return img_byte_arr.getvalue()

#         # 비율 계산
#         width_ratio = max_width / width
#         height_ratio = max_height / height
#         scale = min(width_ratio, height_ratio)
        
#         new_width = int(width * scale)
#         new_height = int(height * scale)

#         # 그레이스케일로 변환
#         if img.mode != 'L':
#             img = img.convert('L')
        
#         img = img.resize((new_width, new_height), Image.LANCZOS)
        
#         img_byte_arr = io.BytesIO()
#         img.save(img_byte_arr, format='PNG')
#         return img_byte_arr.getvalue()
            
#     except Exception as e:
#         exception_list.append(file_path)
#         # logging.error(f"이미지 리사이즈 실패: {str(e)}")
#         return None, exception_list

def resize_image_in_memory(file_path, img, width, height, json_data, exception_list):
    """
    이미지를 메모리에서 리사이즈하는 함수
    """
    try:
        max_width = json_data['ocr_info']['max_width']
        max_height = json_data['ocr_info']['max_height']
        
        # 리사이즈가 필요한지 확인
        if width <= max_width and height <= max_height:
            return process_image(img, file_path)

        # 비율 계산
        width_ratio = max_width / width
        height_ratio = max_height / height
        scale = min(width_ratio, height_ratio)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 이미지 리사이즈
        img = img.resize((new_width, new_height), Image.LANCZOS)
        return process_image(img, file_path)
            
    except Exception as e:
        logging.error(f"이미지 리사이즈 실패: {str(e)}")
        return None
    
def read_tiff_with_conversion(file_path, width, height, json_data, exception_list):
    """
    TIFF 파일을 메모리에서 PNG로 변환하여 OCR 처리
    그레이스케일로 변환하고 필요한 경우 크기를 조정함
    """
    try:
        ocr_info = json_data['ocr_info']
        MAX_WIDTH = ocr_info['max_width']
        MAX_HEIGHT = ocr_info['max_height']
        
        # 이미지 크기가 max_width나 max_height를 초과하는지 미리 확인
        width_ratio = MAX_WIDTH / width
        height_ratio = MAX_HEIGHT / height
        scale = min(width_ratio, height_ratio)
        
        new_width = width
        new_height = height
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            new_width = int(width * scale)
            new_height = int(height * scale)
            logging.info(f"이미지 리사이즈 예정: {file_path} -> {new_width}x{new_height}")

        # PIL.Image.MAX_IMAGE_PIXELS 설정으로 큰 이미지 처리
        if new_width * new_height > Image.MAX_IMAGE_PIXELS:
            Image.MAX_IMAGE_PIXELS = new_width * new_height

        with Image.open(file_path) as img:
            if width > MAX_WIDTH or height > MAX_HEIGHT:
                img = img.resize((new_width, new_height), Image.LANCZOS)
                logging.info(f"이미지 리사이즈 완료: {file_path}")
            
            # TIFF의 첫 페이지만 처리하고 그레이스케일로 변환
            if img.mode != 'L':
                img = img.convert('L')
            
            # 메모리에 PNG로 변환
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue(), exception_list
    except Exception as e:
        exception_list.append(file_path)
        # logging.error(f"TIFF 처리 중 오류 발생 {file_path}: {str(e)}")
        return None, exception_list
    
# def filter_large_images(image_files, json_data, uuid_map, exception_list):
#     filtered_files = []
#     big_size_files = []
    
#     max_pixel_limit = json_data['ocr_info']['max_pixel_limit']
#     tif_mode = json_data['ocr_info']['tif_mode']

#     if tif_mode:
#         logging.info(f"tif mode가 활성화 되어 있음으로 TIFF 이미지 처리를 시작합니다")
#     else:
#         logging.info(f"tif mode가 비활성화 되어 있음으로 TIFF 이미지를 제외한 나머지 이미지 파일 처리를 시작합니다")

#     for file_path in image_files:
#         try:
#             width = height = None
#             if tif_mode and file_path.lower().endswith(('.tif', '.tiff')):
#                 logging.info(f"TIFF 파일 처리 중: {file_path}")
#                 width, height = get_tiff_size_tifffile(file_path, uuid_map, exception_list)
#                 # image_data = read_tiff_with_conversion(file_path, width, height, json_data, uuid_map, exception_list)
#                 if width and height:
#                     if (width * height) > max_pixel_limit:
#                         image_data = read_tiff_with_conversion(file_path, width, height, json_data)
#                     else:
#                         with open(file_path, 'rb') as f:
#                             image_data = f.read()
#             # 다른 이미지 파일 처리
#             else:
#                 width, height = get_image_size_from_metadata(file_path)
#                 if width and height:
#                     try:
#                         with Image.open(file_path) as img:
#                             if (width * height) > max_pixel_limit:
#                                 image_data = resize_image_in_memory(file_path, img, width, height, json_data, big_size_files, uuid_map, exception_list)
#                             else:
#                                 # 이미지를 바로 bytes로 변환
#                                 img_byte_arr = io.BytesIO()
#                                 img.save(img_byte_arr, format='PNG')
#                                 image_data = img_byte_arr.getvalue()
#                     except Exception as e:
#                         handle_failure(file_path, uuid_map, exception_list)
#                         continue

#             if image_data is None:
#                 handle_failure(file_path, uuid_map, exception_list)
#                 continue
#             if image_data:
#                 filtered_files.append((file_path, image_data))
#                 logging.info(f"이미지 필터링 완료: {file_path}")
#         except Exception as e:
#             handle_failure(file_path, uuid_map, exception_list)
#             # logging.error(f"Error reading file {file_path}: {str(e)}")

#     logging.info(f"Filtered files: {len(filtered_files)}")
#     logging.info(f"Big size files: {len(big_size_files)}")


#     return filtered_files, big_size_files
def filter_large_images(image_files, json_data, exception_list):
    """
    이미지 파일을 필터링하고 필요한 경우 리사이즈하는 함수

    Parameters:
        image_files (list): 이미지 파일 경로 리스트
        json_data (dict): 설정 데이터
        exception_list (list): 실패한 파일 목록

    Returns:
        tuple: (처리된 파일 목록, 큰 크기 파일 목록)
    """
    filtered_files = []
    big_size_files = []
    
    # 설정값 가져오기
    max_pixel_limit = json_data['ocr_info']['max_pixel_limit']
    tif_mode = json_data['ocr_info'].get('tif_mode', False)
    
    # logging.info(f"시작: 총 {len(image_files)}개 파일 처리")
    
    for file_path in image_files:
        try:
            # 파일 존재 확인
            if not os.path.isfile(file_path):
                # logging.error(f"파일이 존재하지 않음: {file_path}")
                exception_list.append(file_path)
                continue

            # 파일 확장자 확인
            _, ext = os.path.splitext(file_path.lower())
            
            # TIFF 파일 처리
            if ext in ['.tif', '.tiff']:
                if not tif_mode:
                    logging.debug(f"TIFF 모드 비활성화로 건너뛰기: {file_path}")
                    continue
                    
                width, height = get_tiff_size_tifffile(file_path)
                if not width or not height:
                    logging.error(f"TIFF 크기 읽기 실패: {file_path}")
                    exception_list.append(file_path)
                    continue
                    
                if (width * height) > max_pixel_limit:
                    logging.debug(f"큰 TIFF 파일 변환 시작: {file_path}")
                    image_data, exception_list = read_tiff_with_conversion(file_path, width, height, json_data, exception_list)
                    if image_data:
                        filtered_files.append((file_path, image_data))
                    else:
                        big_size_files.append(file_path)
                else:
                    with open(file_path, 'rb') as f:
                        filtered_files.append((file_path, f.read()))
                        
            # 일반 이미지 파일 처리
            else:
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        if (width * height) > max_pixel_limit:
                            # logging.debug(f"큰 이미지 리사이즈 시작: {file_path}")
                            image_data, exception_list = resize_image_in_memory(file_path, img, width, height, json_data, exception_list)
                            if image_data:
                                filtered_files.append((file_path, image_data))
                            else:
                                big_size_files.append(file_path)
                        else:
                            # 이미지를 PNG로 변환
                            img_byte_arr = io.BytesIO()
                            if img.mode != 'L':
                                img = img.convert('L')
                            img.save(img_byte_arr, format='PNG')
                            filtered_files.append((file_path, img_byte_arr.getvalue()))
                            # logging.debug(f"일반 이미지 처리 완료: {file_path}")
                            
                except Exception as e:
                    # logging.error(f"이미지 처리 실패 {file_path}: {str(e)}")
                    exception_list.append(file_path)
                    
        except Exception as e:
            # logging.error(f"파일 처리 중 오류 발생 {file_path}: {str(e)}")
            exception_list.append(file_path)
            
    # 처리 결과 로깅
    
    return filtered_files, big_size_files, exception_list

# def send_to_ocr_server(batch, json_data, ocr_server_ip, exception_list):
#     """
#     배치 단위로 OCR 서버에 이미지를 전송하는 함수
    
#     Parameters:
#         batch (list): (file_path, image_data) 튜플의 리스트
#         json_data (dict): 설정 데이터
#         ocr_server_ip (str): OCR 서버 IP
        
#     Returns:
#         dict: file_path를 키로 하고 OCR 결과를 값으로 하는 딕셔너리
#     """
#     ocr_info = json_data['ocr_info']
#     ocr_server_port = ocr_info['ocr_server_port']
#     url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"
#     logging.info(f"OCR 서버 전송 시작")
#     results = {}
#     for file_path, image_data in batch:
#         try:
#             response = requests.post(url, files={'image': image_data})
#             results[file_path] = response.json()
#         except Exception as e:
#             exception_list.append(file_path)
            
#     return results, exception_list

async def send_batch_to_server(session, url, batch_data, uuid_map):
    """
    배치 데이터를 OCR 서버에 비동기로 전송하는 함수
    
    Parameters:
        session (aiohttp.ClientSession): 비동기 HTTP 세션
        url (str): OCR 서버 URL
        batch_data (list): (file_path, image_data) 튜플의 리스트
        uuid_map (dict): 파일 경로와 UUID 매핑
        
    Returns:
        list: OCR 결과 리스트
    """
    tasks = []
    for file_path, image_data in batch_data:
        uuid_str = uuid_map[file_path]
        
        language = detect_language(file_path)
        # FormData 생성
        form_data = aiohttp.FormData()
        form_data.add_field('file', image_data)
        form_data.add_field('language', language)
        form_data.add_field('uuid_str', uuid_str)
        
        # 비동기 태스크 생성
        task = asyncio.create_task(send_single_image(session, url, form_data, file_path, uuid_str))
        tasks.append(task)
    
    # 모든 태스크 완료 대기
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def send_single_image(session, url, form_data, file_path, uuid_str):
    """
    단일 이미지를 OCR 서버에 전송하는 함수
    """
    try:
        async with session.post(url, data=form_data) as response:
            if response.status == 200:
                result = await response.json()
                # 응답 형식 확인
                if isinstance(result, dict) and 'result' in result:
                    return {
                        'file_path': file_path,
                        'uuid_str': uuid_str,
                        'result': result['result'],
                        'success': True
                    }
            # logging.error(f"서버 응답 오류 - status: {response.status}, file: {file_path}")
            return {
                'file_path': file_path,
                'uuid_str': uuid_str,
                'result': None,
                'success': False
            }
    except asyncio.TimeoutError:
        logging.error(f"서버 요청 시간 초과: {file_path}")
        return {
            'file_path': file_path,
            'uuid_str': uuid_str,
            'result': None,
            'success': False
        }
    except Exception as e:
        logging.error(f"이미지 전송 중 오류 발생 {file_path}: {str(e)}")
        return {
            'file_path': file_path,
            'uuid_str': uuid_str,
            'result': None,
            'success': False
        }

# async def process_batch_async(batch_data, json_data, ocr_server_ip, ocr_server_port, uuid_map):
#     """
#     배치 데이터를 비동기로 처리하는 함수
#     """
#     url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"
    
#     async with aiohttp.ClientSession() as session:
#         results = await send_batch_to_server(session, url, batch_data, uuid_map)
        
#         # 결과 처리
#         for result in results:
#             if result['success']:
#                 save_as_json(
#                     result['file_path'],
#                     result['uuid_str'],
#                     result['result'],
#                     json_data,
#                     True
#                 )
#             else:
#                 save_as_json(
#                     result['file_path'],
#                     result['uuid_str'],
#                     None,
#                     json_data,
#                     False
#                 )
                
#     return results

async def process_batch_async(batch_data, json_data, ocr_server_ip, ocr_server_port, uuid_map):
    """
    배치 데이터를 비동기로 처리하는 함수
    """
    url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for file_path, image_data in batch_data:
            uuid_str = uuid_map[file_path]
            
            language = detect_language(file_path)
            
            # FormData 생성
            form_data = aiohttp.FormData()
            form_data.add_field('file', image_data)
            form_data.add_field('language', language)
            form_data.add_field('uuid_str', uuid_str)
            
            task = send_single_image(session, url, form_data, file_path, uuid_str)
            tasks.append(task)
        
        # 배치의 모든 요청을 동시에 처리
        batch_results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            batch_results.append(result)
        
        # 각 결과에 대해 즉시 JSON 생성
        for result in batch_results:
            file_path = result['file_path']
            uuid_str = result['uuid_str']
            success = result['success']
            
            if success:
                # OCR 성공 시 JSON 생성
                save_as_json(
                    file_path=file_path,
                    uuid_str=uuid_str,
                    response=result['result'],
                    json_data=json_data,
                    success=True
                )
            else:
                # OCR 실패 시 실패 정보로 JSON 생성
                save_as_json(
                    file_path=file_path,
                    uuid_str=uuid_str,
                    response={},
                    json_data=json_data,
                    success=False
                )
            
            results.append(result)
            
            # 진행 상황 로깅
            # logging.info(f"처리 완료 - 파일: {file_path}, 성공 여부: {'성공' if success else '실패'}")
            
    return results

# def process_batch_in_process(args):
#     """
#     각 프로세스에서 실행될 배치 처리 함수
#     """
#     batch_data, json_data, ocr_server_ip, ocr_server_port, uuid_map = args
    
#     # 비동기 이벤트 루프 생성 및 실행
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
    
#     try:
#         results = loop.run_until_complete(
#             process_batch_async(batch_data, json_data, ocr_server_ip, ocr_server_port, uuid_map)
#         )
#         return results
#     finally:
#         loop.close()

def process_batch_in_process(args):
    """
    각 프로세스에서 실행될 배치 처리 함수
    """
    batch_data, json_data, ocr_server_ip, ocr_server_port, uuid_map = args
    
    # 비동기 이벤트 루프 생성 및 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(
            process_batch_async(batch_data, json_data, ocr_server_ip, ocr_server_port, uuid_map)
        )
        
        # 배치 처리 결과 로깅
        success_count = sum(1 for r in results if r['success'])
        failure_count = len(results) - success_count
        # logging.info(f"배치 처리 완료 - 성공: {success_count}, 실패: {failure_count}")
        
        return results
    finally:
        loop.close()

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
        else:
            ocr_data["tags"] = ["ocr", "file", "N", "exception"]
    
    except Exception as e:
        error_message = f"오류 발생: {str(e)}"
        response = {"exception_message": error_message, "tags" : ["ocr", "file", "N", "exception"]}

    json_file_name = os.path.join(result_path, f"{uuid_str}.json")

    try:
        # with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_string = json.dumps(ocr_data, ensure_ascii=False, indent=4) 
    except Exception as e:
        error_message = f"JSON 구조 생성 중 알 수 없는 오류 발생 : {str(e)}"
        logging.error(error_message)
    else:
        try:
            with open(json_file_name, 'w', encoding='utf-8') as json_file:
                json_file.write(json_string)
        except Exception as e:
            error_message = f"JSON 파일 생성 중 알 수 없는 오류 발생 : {str(e)}"
            logging.error(error_message)

    return json_failed_count

def main():
    global success_count, failure_count
    
    json_data = config_reading('config.json')
    if not json_data:
        return

    try:
        root_path = json_data['root_path']
        datainfopath = json_data['datainfopath']
        source_path = datainfopath['target_path']
        source_path = os.path.join(root_path, source_path)

        datafilter = json_data['datafilter']
        image_extensions = datafilter['image_extensions']

        ocr_info = json_data['ocr_info']
        batch_size = ocr_info["batch_size"]
        ocr_server_ip = sys.argv[1] 
        ocr_server_port = ocr_info["ocr_server_port"]
        process_count = ocr_info["process_count"]
        log_to_console = ocr_info["log_to_console"]
        log_to_file = ocr_info["log_to_file"]
        log_to_level = ocr_info["log_to_level"]

        current_directory = os.getcwd()
        current_time = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(current_directory, f"ocr_processor_{current_time}.log")
        setup_logging(log_to_console, log_to_file, log_file_path, log_to_level)

    except Exception as e:
        logging.error(f"Error reading config.json: {str(e)}")
        return

    # 이미지 파일 수집
    # image_files = [
    #     get_safe_path(os.path.join(root, file))
    #     for root, _, files in os.walk(source_path)
    #     for file in files
    #     if any(file.lower().endswith(ext) for ext in image_extensions)
    # ]
    image_files = []
    for root, _, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)

    # UUID 매핑 초기화
    uuid_map = initialize_uuid_map(image_files)
    exception_list = []

    # 이미지 필터링을 위한 배치 생성
    filter_batch_size = 2000  # 필터링용 배치 크기
    filter_batches = [
        image_files[i:i + filter_batch_size]
        for i in range(0, len(image_files), filter_batch_size)
    ]

    # 멀티프로세싱 설정
    num_processes = min(cpu_count(), process_count)

    # 이미지 필터링 멀티프로세스 처리
    filtered_files = []
    big_size_files = []
    with Pool(processes=num_processes) as pool:
        filter_args = [(batch, json_data, exception_list) for batch in filter_batches]
        filter_results = pool.starmap(filter_large_images, filter_args)
        
        for result in filter_results:
            filtered, big_files, exceptions = result
            filtered_files.extend(filtered)
            big_size_files.extend(big_files)
            exception_list.extend(exceptions)
            
    logging.info(f"파일 전처리 완료: 성공: {len(filtered_files)}개, 큰 파일: {len(big_size_files)}개, 실패: {len(exception_list)}개")

    # OCR 처리를 위한 배치 생성
    start_time = time.time()
    ocr_batches = [
        filtered_files[i:i + batch_size]
        for i in range(0, len(filtered_files), batch_size)
    ]
    
    # OCR 프로세스 풀 생성 및 작업 분배
    with Pool(processes=num_processes) as pool:
        process_args = [
            (batch, json_data, ocr_server_ip, ocr_server_port, uuid_map)
            for batch in ocr_batches
        ]
        
        all_results = []
        for batch_results in pool.imap_unordered(process_batch_in_process, process_args):
            all_results.extend(batch_results)
            
            # 현재까지의 진행 상황 출력
            current_success = sum(1 for r in all_results if r['success'])
            current_failure = len(all_results) - current_success
            # logging.info(f"진행 상황 \n 성공: {current_success}, 실패: {current_failure}")
    
    # 실패 파일 처리
    process_exceptions(exception_list, uuid_map, json_data)

    # 결과 출력
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logging.info(f"총 이미지 파일 개수: {len(image_files)}")
    logging.info(f"총 분석 시간: {hours}시간, {minutes}분, {seconds:.2f}초")
    logging.info(f"분석 성공 파일: {success_count}개, 분석 실패 파일: {len(exception_list)}개")


if __name__ == "__main__":
    main()