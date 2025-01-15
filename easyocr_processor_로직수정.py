from logging.handlers import RotatingFileHandler
import os
import logging
import time
import json
import datetime
import uuid
import io
import sys
import magic
import struct
import aiohttp
from PIL import Image
import asyncio
from langdetect import detect
import pytesseract
from multiprocessing import Pool, Manager, cpu_count
import math
import cv2
import numpy as np
from pillow_heif import register_heif_opener
import aiofiles
from multiprocessing import Lock

json_lock = Lock()

register_heif_opener()


tiff_success = 0
tiff_failure = 0
img_success = 0
img_failure = 0
json_failed_count = 0
current_time = time.strftime("%Y%m%d%H%M%S")
ocr_ver = 1.0

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

    # PIL 라이브러리에서 나오는 경고 수준 로그를 억제
    # logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("libpng").setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*libpng.*')
    warnings.filterwarnings('ignore', message='.*iCCP.*')
    
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

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

def generate_uuid_str():
    """UUID와 타임스탬프를 조합하여 고유 문자열 생성"""
    new_uuid = uuid.uuid4()
    utc_now = datetime.datetime.utcnow()
    timestamp = utc_now.timestamp()
    uuid_str = f"{new_uuid}_{timestamp}"
    return uuid_str

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

def get_main_fold(target_path, file_path):
    """
    파일 경로에서 주 폴더를 가져오는 함수
    
    Parameters:
        target_path (str): 대상 경로
        file_path (str): 파일 경로
    
    Returns:
        str: 주 폴더 또는 None (폴더를 찾을 수 없는 경우)
    """
    relative_path = os.path.relpath(file_path, target_path)
    components = os.path.normpath(relative_path).split(os.path.sep)
    main_fold = components[0] if len(components) > 1 else None
    
    if main_fold:
        parts = main_fold.split('_')
        filtered_parts = [part for part in parts if not part.isdigit()]
        main_fold = '_'.join(filtered_parts)
    
    return main_fold

def get_jpeg_size(file_path):
    """
    JPEG 파일의 해상도(가로, 세로 픽셀)를 헤더에서 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    with open(file_path, 'rb') as f:
        f.seek(0)
        f.read(2)  # SOI (Start of Image) 부분 건너뛰기
        while True:
            byte = f.read(1)
            while byte != b'\xFF':
                byte = f.read(1)
            marker = f.read(1)
            if marker in [b'\xC0', b'\xC2']:  # SOF0 or SOF2 (Start of Frame markers)
                f.read(3)
                height, width = struct.unpack(">HH", f.read(4))
                return width, height
            else:
                size = struct.unpack(">H", f.read(2))[0]
                f.read(size - 2)

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

def convert_heic_to_png(file_path):
    """
    HEIC 파일을 PNG로 변환하는 함수
    
    Parameters:
        file_path (str): HEIC 파일 경로
    
    Returns:
        bytes: PNG 형식의 이미지 데이터
    """
    try:
        # HEIC 파일 열기
        with Image.open(file_path) as heic_image:
            # RGB로 변환
            if heic_image.mode != 'RGB':
                heic_image = heic_image.convert('RGB')
            
            # PNG로 변환
            output = io.BytesIO()
            heic_image.save(output, format='PNG')
            return output.getvalue()
            
    except Exception as e:
        logging.error(f"Error converting HEIC file {file_path}: {str(e)}")
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

def get_image_size_from_metadata(file_path):
    """
    지원되는 파일 확장자에 대해 이미지의 크기를 메타데이터에서 추출하는 함수
    
    Parameters:
        file_path (str): 파일 경로
    
    Returns:
        tuple: 가로, 세로 픽셀 크기
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if not is_file_accessible(file_path):
        return None, None

    if ext in ['.jpg', '.jpeg']:
        return get_jpeg_size(file_path)
    elif ext == '.png':
        return get_png_size(file_path)
    elif ext == '.gif':
        return get_gif_size(file_path)
    elif ext == '.bmp':
        return get_bmp_size(file_path)
    elif ext in ['.tiff', '.tif']:
        return get_tiff_size(file_path)
    elif ext in ['.emf', '.wmf']:
        # logging.error(f"Unsupported file type for {file_path}")
        return None, None
    else:
        # Tesseract가 지원하지 않는 형식의 이미지
        # logging.info(f"Unsupported file type for {file_path}")
        return None, None

def convert_gif_to_png(file_path):
    """
    GIF 파일을 PNG로 변환하는 함수
    
    Parameters:
        file_path (str): GIF 파일 경로
    
    Returns:
        numpy.ndarray: OpenCV 이미지 배열
    """
    try:
        # PIL로 GIF 파일 읽기
        with Image.open(file_path) as img:
            # 첫 번째 프레임만 사용
            if hasattr(img, 'n_frames'):
                img.seek(0)
            
            # RGBA인 경우 RGB로 변환
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # PIL 이미지를 numpy 배열로 변환
            img_array = np.array(img)
            # RGB to BGR (OpenCV 형식으로 변환)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
            
    except Exception as e:
        logging.error(f"Error converting GIF file {file_path}: {str(e)}")
        return None

def detect_language(image_data):
    """
    이미지에서 텍스트의 언어를 감지하는 함수
    
    Parameters:
        image_data (bytes): 이미지 데이터
    
    Returns:
        str: 감지된 언어 코드 ('en', 'ko', 'ja', 'ch_sim')
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        # 한국어, 영어, 일본어, 중국어(간체) OCR
        text = pytesseract.image_to_string(image, lang='eng+kor+jpn+chi_sim')
        
        if not text or len(text.strip()) < 5:
            return "en"
        
        try:
            detected = detect(text)
            if isinstance(detected, set):
                detected = 'en'
            
            # 언어 매핑 정의
            language_mapping = {
                'ko': 'ko',
                'ja': 'ja',
                'zh-cn': 'ch_sim',
                'zh-tw': 'ch_sim',
                'en': 'en'
            }
            
            # 문자 범위 체크를 위한 함수들
            def has_korean(text):
                return any(ord('가') <= ord(c) <= ord('힣') for c in text)
                
            def has_japanese(text):
                return any((ord('ぁ') <= ord(c) <= ord('ゖ')) or 
                          (ord('ァ') <= ord(c) <= ord('ヺ')) for c in text)
                
            def has_chinese(text):
                return any(ord('一') <= ord(c) <= ord('龯') for c in text)
            
            # 영문 비율 계산
            total_len = len(text)
            eng_ratio = len([c for c in text if c.isascii()]) / total_len if total_len > 0 else 0
            
            # 언어 판별 로직 개선
            if detected in language_mapping:
                return language_mapping[detected]
            elif has_korean(text):
                return 'ko'
            elif has_japanese(text):
                return 'ja'
            elif has_chinese(text):
                return 'ch_sim'
            elif eng_ratio > 0.5:
                return 'en'
            else:
                return 'ko'  # 기본값으로 한국어 반환
                
        except Exception as e:
            return "ko"
            
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        return "ko"

def process_and_resize_image(file_path, json_data):
    """
    이미지 파일의 크기를 확인하고, 리사이즈하거나 처리 불가 상태를 반환함
    
    Parameters:
        file_path (str): 이미지 파일 경로
        json_data (dict): 설정 데이터

    Returns:
        bytes: 리사이즈된 이미지 데이터
    """
    global img_failure
    max_width = json_data['ocr_info']['max_width']
    max_height = json_data['ocr_info']['max_height']
    max_pixel_limit = json_data['ocr_info']['max_pixel_limit']

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width * height > max_pixel_limit:
                new_uuid = uuid.uuid4()
                utc_now = datetime.datetime.utcnow()
                timestamp = utc_now.timestamp()
                uuid_str = f"{new_uuid}_{timestamp}"
                save_as_json(file_path, uuid_str, {}, json_data, success=False)
                # logging.warning(f"이미지가 너무 큽니다: {fi/_path} ({width}x{height} 픽셀)")
                return None

            if width > max_width and height > max_height:
                # logging.info(f"이미지 가로 세로가 커서 리사이즈 시작")
                scale = min(max_width / width, max_height / height)
                new_width, new_height = int(width * scale), int(height * scale)
                img = img.resize((new_width, new_height), Image.ANTIALIAS)

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
    except Exception as e:
        # logging.error(f"이미지 처리 중 오류 발생: {file_path}, {str(e)}")
        img_failure += 1
        return None

# 이미지 그레이스케일로 변환
def process_image_to_grayscale(image_data: bytes) -> bytes:
    """
    이미지를 그레이스케일로 변환하는 함수
    
    Parameters:
        image_data (bytes): 이미지 데이터
    
    Returns:
        bytes: 그레이스케일로 변환된 이미지 데이터
    """
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            # TIFF 파일인 경우 첫 페이지만 처리
            if image.format.lower() == 'TIFF' or image.format.lower() == 'TIF':
                try:
                    img.seek(0)
                except EOFError:
                    pass
                
                # TIFF 파일 모드 처리
                if img.mode == 'P':
                    img = img.convert('RGB')
                elif img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode not in ['L', 'RGB']:
                    img = img.convert('RGB')
            
            # 그레이스케일 변환
            grayscale_image = image.convert("L")
            output = io.BytesIO()
            grayscale_image.save(output, format='PNG')
            return output.getvalue()
    except Exception as e:
        # logging.error(f"Error converting image to grayscale: {str(e)}")
        return image_data

import warnings

def filter_image_files(file_list):
    """
    이미지 파일 필터링 함수
    - macOS 메타데이터 파일 제외
    - __MACOSX 디렉토리 제외
    """
    filtered_files = []
    for file_path in file_list:
        # macOS 메타데이터 및 시스템 파일 제외
        if (os.path.basename(file_path).startswith('._') or 
            '__MACOSX' in file_path or 
            os.path.basename(file_path).startswith('.')):
            continue
        
        # 실제 파일 존재 여부 확인
        if not os.path.isfile(file_path):
            continue
            
        filtered_files.append(file_path)
    return filtered_files

def filter_large_images(image_files, json_data):
    """
    대용량 이미지 필터링 함수

    Parameters:
        image_files (list): 이미지 파일 경로 리스트
        json_data (dict): 설정 데이터
    
    Returns:
        tuple: 필터링된 이미지 파일 리스트와 대용량 이미지 파일 리스트
    """
    global img_failure, img_success
    filtered_files = []
    big_size_files = []
    
    max_pixel_limit = json_data['ocr_info']['max_pixel_limit']
    for file_path in image_files:
        try:
            image_data = process_image_to_grayscale(file_path)
            image_data = process_and_resize_image(file_path, json_data)

            if image_data is None:
                img_failure += 1
                continue
            else:
                width, height = get_image_size_from_metadata(file_path)
                uuid_str = generate_uuid_str()
                if width is not None and height is not None:
                    if (width * height) > max_pixel_limit:
                        body_info = {}
                        save_as_json(file_path, uuid_str, body_info, json_data, success=False)
                        big_size_files.append(file_path)
                    else:
                        # logging.info(f"이미지 크기 적절한 파일들 처리 시작")
                        filtered_files.append((file_path, image_data))
                else:
                    body_info = {}
                    img_failure += 1
                    save_as_json(file_path, uuid_str, body_info, json_data, success=False)
        except Exception as e:
            uuid_str = generate_uuid_str()
            logging.error(f"Error reading file {file_path}: {str(e)}")
            body_info = {} 
            img_failure += 1
            save_as_json(file_path, uuid_str, body_info, json_data, success=False)

    return filtered_files, big_size_files

# 20241120 주석처리
# async def send_image(session, url, file_path, image_data, language, retry_count=3):
#     """
#     이미지를 서버로 전송하는 함수. 실패 시 최대 retry_count만큼 재시도.
#     """
#     uuid_str = generate_uuid_str()
    
#     for attempt in range(retry_count):
#         try:
#             form_data = aiohttp.FormData()
#             form_data.add_field('file', image_data)
#             form_data.add_field('language', language)
#             form_data.add_field('uuid_str', uuid_str)
            
#             async with session.post(url, data=form_data) as response:
#                 if response.status == 200:
#                     return file_path, uuid_str, await response.json(), True
#                 else:
#                     logging.warning(f"서버 응답 오류: {response.status}, 파일: {file_path}")
#         except Exception as e:
#             logging.error(f"이미지 전송 중 오류 발생 (시도 {attempt+1}/{retry_count}): {str(e)}")
#             if attempt == retry_count - 1:  # 마지막 시도였다면
#                 return file_path, uuid_str, None, False
#             await asyncio.sleep(1)  # 재시도 전에 대기
    
#     return file_path, uuid_str, None, False

async def send_image(session, url, uuid_str, image_data, language):
    """
    이미지와 UUID를 서버에 전송
    
    Parameters:
        session (aiohttp.ClientSession): 클라이언트 세션
        url (str): OCR 서버 URL
        uuid_str (str): UUID 문자열
        image_data (bytes): 이미지 데이터
        language (str): 언어 코드
    """
    try:
        if not isinstance(image_data, bytes):
            logging.error(f"잘못된 이미지 데이터 형식: {type(image_data)}")
            return uuid_str, None, False
        form_data = aiohttp.FormData()
        # form_data.add_field("file", image_data)
        form_data.add_field(image_data)
        form_data.add_field("uuid", uuid_str)
        form_data.add_field("language", language)

        async with session.post(url, data=form_data) as response:
            if response.status == 200:
                return uuid_str, await response.json(), True
            else:
                logging.warning(f"서버 응답 오류: {response.status}")
    except Exception as e:
        logging.error(f"이미지 전송 실패: {str(e)}")
    return uuid_str, None, False


def resize_image_tiff(image, json_data):
    """
    이미지 크기를 제한에 맞게 조정하는 함수
    
    Parameters:
        image (PIL.Image): PIL 이미지 객체
        json_data (dict): 설정 데이터
    
    Returns:
        PIL.Image: 리사이즈된 이미지 객체
    """
    max_width = json_data['ocr_info']['max_width']
    max_height = json_data['ocr_info']['max_height']
    max_pixel_limit = json_data['ocr_info']['max_pixel_limit']

    width, height = image.size
    if width * height > max_pixel_limit or width > max_width or height > max_height:
        image.thumbnail((max_width, max_height))

    return image


# def process_tiff_file(file_path, json_data):
#     with Image.open(file_path) as img:
#         n_frames = getattr(img, 'n_frames', 1)
#         for page in range(n_frames):
#             try:
#                 img.seek(page)
#                 frame = img.convert("RGB")
#                 yield frame
#             except Exception as e:
#                 logging.error(f"Error processing TIFF page {page} in {file_path}: {str(e)}")
#                 continue

def process_tiff_file(file_path, json_data, shared_counters):
    """
    TIFF 파일의 첫 번째 페이지를 처리
    
    Parameters:
        file_path (str): TIFF 파일 경로
        json_data (dict): 설정 데이터
        shared_counters (dict): 공유 카운터
    """
    try:
        with Image.open(file_path) as img:
            img.seek(0)  # 첫 번째 페이지만 사용
            frame = img.convert("RGB")
            return frame
    except Exception as e:
        # logging.error(f"TIFF 파일 처리 중 오류 발생: {file_path}, {str(e)}")
        return None



async def send_tiff_pages_to_server(session, ocr_server_ip, ocr_server_port, uuid_str, pages_data, language):
    """
    TIFF 파일의 모든 페이지를 서버에 전송하고 결과를 합치는 함수
    
    Parameters:
        session (aiohttp.ClientSession): 클라이언트 세션
        ocr_server_ip (str): OCR 서버 IP 주소
        ocr_server_port (str): OCR 서버 포트
        uuid_str (str): UUID 문자열
        pages_data (bytes): TIFF 페이지 데이터
        language (str): 언어 코드
    """
    all_texts = []
    url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"

    # 수정된 부분: pages_data는 이제 단일 바이트 객체
    try:
        form_data = aiohttp.FormData()
        form_data.add_field('file', pages_data)
        form_data.add_field('language', language)
        form_data.add_field('uuid_str', uuid_str)
        
        async with session.post(url, data=form_data) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, str):
                    try:
                        result = await response.json()
                    except:
                        result = {"text": result}
                return uuid_str, result, True
            return uuid_str, result, True
    except Exception as e:
        logging.error(f"Error processing TIFF for UUID {uuid_str}: {str(e)}")
        return uuid_str, None
  
# 20241120 주석처리
# async def process_image_batch(batch_files, json_data, ocr_server_ip, ocr_server_port, shared_counters):
#     """
#     배치 단위로 이미지를 처리하는 함수
    
#     Parameters:
#         batch_files (list): 이미지 파일 경로 리스트
#         json_data (dict): 설정 데이터
#         ocr_server_ip (str): OCR 서버 IP 주소
#         ocr_server_port (str): OCR 서버 포트
#         shared_counters (dict): 공유 카운터
#     """
#     try:
#         async with aiohttp.ClientSession() as session:
#             tasks = []
#             uuid_to_file_map = {}

#             for file_path in batch_files:
#                 if isinstance(file_path, tuple):
#                     file_path, image_data = file_path
#                 else:
#                     image_data = None

#                 new_uuid = uuid.uuid4()
#                 utc_now = datetime.datetime.utcnow()
#                 timestamp = utc_now.timestamp()
#                 uuid_str = f"{new_uuid}_{timestamp}"
#                 uuid_to_file_map[uuid_str] = file_path

#                 if image_data is None:
#                     with open(file_path, 'rb') as f:
#                         image_data = f.read()

#                 language = detect_language(image_data)
#                 if language is None:
#                     language = "unknown"

#                 task = asyncio.ensure_future(send_image(session, f"http://{ocr_server_ip}:{ocr_server_port}/ocr/", uuid_str, image_data, language))
#                 tasks.append(task)

#             for task in asyncio.as_completed(tasks):
#                 try:
#                     uuid_str, response = await task
#                     file_path = uuid_to_file_map.get(uuid_str)
#                     if file_path:
#                         success = True if response else False
#                         save_as_json(file_path, uuid_str, response, json_data, success)
                        
#                         # 공유 카운터 업데이트
#                         if success:
#                             shared_counters['success_count'] += 1
#                         else:
#                             shared_counters['failure_count'] += 1
                            
#                 except Exception as e:
#                     logging.error(f"응답 처리 중 오류 발생: {str(e)}")
#                     shared_counters['failure_count'] += 1

#     except Exception as e:
#         logging.error(f"배치 처리 중 오류 발생: {str(e)}")

# 20241120 수정
async def process_image_batch(batch_files, json_data, ocr_server_ip, ocr_server_port, uuid_to_file_map, shared_counters):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for uuid_str, file_path in batch_files:
            async with aiofiles.open(file_path, mode='rb') as f:
                image_data = await f.read()

            language = detect_language(image_data)
            tasks.append(send_image(session, f"http://{ocr_server_ip}:{ocr_server_port}/ocr/", uuid_str, image_data, language))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                shared_counters["failure_count"] += 1
                continue
            uuid_str, response, success = result
            file_path = uuid_to_file_map.get(uuid_str)
            save_as_json(file_path, uuid_str, response, json_data, success)


async def process_batch_async(batch, json_data, ocr_server_ip, ocr_server_port, uuid_to_file_map):
    """
    비동기적으로 배치를 처리하고 UUID와 서버 응답 매칭
    
    Parameters:
        batch (list): 이미지 파일 경로와 UUID 매핑 리스트
        json_data (dict): 설정 데이터
        ocr_server_ip (str): OCR 서버 IP 주소
        ocr_server_port (str): OCR 서버 포트
        uuid_to_file_map (dict): UUID와 파일 경로 매핑 딕셔너리
    """
    global img_failure, img_success
    url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for uuid_str, file_path in batch:
            try:
                # 이미지 파일을 바이너리 모드로 읽기
                async with aiofiles.open(file_path, mode='rb') as f:
                    image_data = await f.read()
                
                # 이미지 데이터 유효성 검사
                if not image_data:
                    # raise ValueError(f"빈 이미지 데이터: {file_path}")
                    img_failure += 1
                    save_as_json(file_path, uuid_str, {}, json_data, success=False)
                    continue
                
                language = detect_language(image_data)
                tasks.append(send_image(session, url, uuid_str, image_data, language))
                
            except Exception as e:
                # logging.error(f"이미지 준비 실패 ({file_path}): {str(e)}")
                img_failure += 1
                save_as_json(file_path, uuid_str, {}, json_data, success=False)

        # 서버 응답 처리
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            try:
                if isinstance(result, Exception):
                    # logging.error(f"서버 처리 오류: {result.__class__.__name__}")
                    img_failure += 1
                    continue
                    
                uuid_str, response, success = result
                file_path = uuid_to_file_map.get(uuid_str)
                
                if success:
                    # logging.info(f"이미지 처리 성공: {file_path}")
                    img_success += 1
                    save_as_json(file_path, uuid_str, response, json_data, success=True)
                else:
                    # logging.error(f"이미지 처리 실패: {file_path}")
                    img_failure += 1
                    save_as_json(file_path, uuid_str, {}, json_data, success=False)
                    
            except Exception as e:
                # logging.error(f"결과 처리 오류: {str(e)}")
                img_failure += 1
                if 'file_path' in locals() and 'uuid_str' in locals():
                    save_as_json(file_path, uuid_str, {}, json_data, success=False)
                    
# async def process_batch_async(batch, json_data, ocr_server_ip, ocr_server_port):
#     """
#     비동기적으로 이미지 배치를 처리하며 세션을 재사용.
#     """
#     url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"
    
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for file_path, image_data in batch:
#             try:
#                 uuid_str = generate_uuid_str()
#                 language = detect_language(image_data)
#                 tasks.append(send_image(session, url, uuid_str, image_data, language))
#             except Exception as e:
#                 logging.error(f"이미지 처리 준비 중 오류 발생: {file_path}, {str(e)}")
#                 uuid_str = generate_uuid_str()
#                 save_as_json(file_path, uuid_str, {}, json_data, success=False)
#                 continue

#         # 결과 처리
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#         for i, result in enumerate(results):
#             try:
#                 file_path = batch[i][0]
#                 if isinstance(result, Exception):
#                     logging.error(f"이미지 처리 중 오류 발생: {file_path}, {str(result)}")
#                     uuid_str = generate_uuid_str()
#                     save_as_json(file_path, uuid_str, {}, json_data, success=False)
#                     continue

#                 uuid_str, response, success = result
#                 save_as_json(file_path, uuid_str, response, json_data, success)
#             except Exception as e:
#                 logging.error(f"결과 처리 중 오류 발생: {str(e)}")
#                 # 예외 발생 시에도 JSON 생성
#                 try:
#                     file_path = batch[i][0]
#                     uuid_str = generate_uuid_str()
#                     save_as_json(file_path, uuid_str, {}, json_data, success=False)
#                 except:
#                     logging.error(f"JSON 생성 실패 처리 중 오류 발생")

# 20241120 주석처리
# async def process_batch_async(batch, json_data, ocr_server_ip, ocr_server_port):
#     """
#     비동기적으로 이미지 배치를 처리하며 세션을 재사용.
#     """
#     url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"
    
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for file_path, image_data in batch:
#             try:
#                 language = detect_language(image_data)
#                 tasks.append(send_image(session, url, file_path, image_data, language))
#             except Exception as e:
#                 logging.error(f"이미지 처리 준비 중 오류 발생: {file_path}, {str(e)}")
#                 uuid_str = generate_uuid_str()
#                 save_as_json(file_path, uuid_str, {}, json_data, success=False)
#                 continue

#         # 결과 처리
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#         processed_results = []
        
#         # file_path와 uuid를 매핑하는 딕셔너리 생성
#         file_uuid_map = {result[0]: result[1] for file_path, image_data in batch}
        
#         for result in results:
#             try:
#                 if isinstance(result, Exception):
#                     logging.error(f"이미지 처리 중 오류 발생: {str(result)}")
#                     continue
                
#                 file_path, uuid_str, response, success = result
                
#                 # 서버에서 받은 uuid로 file_path 매칭
#                 if file_path in file_uuid_map:
#                     save_as_json(file_path, uuid_str, response, json_data, success)
#                     processed_results.append(success)
#                 else:
#                     logging.error(f"매칭되는 파일을 찾을 수 없음: {uuid_str}")
#                     processed_results.append(False)
                
#             except Exception as e:
#                 logging.error(f"결과 처리 중 오류 발생: {str(e)}")
#                 if 'file_path' in locals():
#                     uuid_str = generate_uuid_str()
#                     save_as_json(file_path, uuid_str, {}, json_data, success=False)
#                     processed_results.append(False)

#         return processed_results

def save_as_json(file_path, uuid_str, response, json_data, success):
    global json_failed_count
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
            # OCR 서버 응답이 문자열인지, 딕셔너리인지 확인
            if isinstance(response, str):
                ocr_text = response.get("text", "").strip()
            elif isinstance(response, dict):
                ocr_text = response.get("text", "").strip()
            
            if ocr_text:
                ocr_data["content"] = ocr_text
                ocr_data["summary"] = ocr_text[:300]  # 처음 300자를 summary로
                ocr_data["tags"] = ["ocr", "file", "S"]
            else:
                ocr_data["tags"] = ["ocr", "file", "N", "exception"]
        else:
            ocr_data["tags"] = ["ocr", "file", "N", "exception"]
            
    except Exception as e:
        logging.error(f"JSON 생성 중 오류 발생: {str(e)}")
        ocr_data["tags"] = ["ocr", "file", "N", "exception"]


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
            with json_lock:
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
    global tiff_success, tiff_failure, img_success, img_failure
    json_data = config_reading('config.json')

    manager = Manager()
    shared_counters = manager.dict({
        "tiff_success": 0,
        "tiff_failure": 0,
        "img_success": 0,
        "img_failure": 0,
    })

    try:
        if json_data is not None:
            root_path = json_data['root_path']
            datainfopath = json_data['datainfopath']
            source_path = datainfopath['source_path']

            source_path = os.path.join(root_path, source_path)

            datafilter = json_data['datafilter']
            image_extensions = datafilter['image_extensions']

            ocr_info = json_data['ocr_info']
            batch_size = ocr_info["batch_size"]
            ocr_server_port = ocr_info["ocr_server_port"]
            process_count = ocr_info["process_count"]
            tif_mode = ocr_info["tif_mode"]

            log_to_console = ocr_info["log_to_console"]
            log_to_console = ocr_info["log_to_console"]
            log_to_level = ocr_info["log_to_level"]
            log_to_file = ocr_info["log_to_file"]

            current_directory = os.getcwd()
            current_time = time.strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(current_directory, f"ocr_processor_{current_time}.log")
            setup_logging(log_to_console, log_to_file, log_file_path, log_to_level)
            logging.getLogger('PIL').setLevel(logging.ERROR)
            warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
            logging.info(f"ocr_processor ver {ocr_ver}")
    except Exception as e:
        error_log = f"config.json 을 읽는 도중 오류 발생 : {str(e)}"
        logging.error(f"{error_log}")        
        return
    
    ocr_server_ip = sys.argv[1]
    success = False
    logging.info(f"OCR START")

    start_time = time.time()
    image_files = []

    # 파일 분류
    tiff_files, other_image_files, exception_files = [], [], []
    # 파일 분류 수정
    for root, _, files in os.walk(source_path):
        for file in files:
            
            file_path = os.path.join(root, file)
            image_files.append(file_path)  # 전체 파일 개수 추적
            if (os.path.basename(file_path).startswith('._') or '__MACOSX' in file_path or os.path.basename(file_path).startswith('.')):
                exception_files.append(file_path)
                new_uuid = generate_uuid_str()
                save_as_json(file_path, new_uuid, {}, json_data, success=False)
                img_failure += 1
                continue
            if file_path.lower().endswith(('.tif', '.tiff')):
                logging.info(f"{file_path} TIFF 파일")
                tiff_files.append(file_path)
            elif file_path.lower().endswith(('.bmp', ".emf", ".wmf")):
                new_uuid = generate_uuid_str()
                save_as_json(file_path, new_uuid, {}, json_data, success=False)
                img_failure += 1
            elif any(file_path.lower().endswith(ext) for ext in image_extensions):
                other_image_files.append(file_path)
                
    logging.info(f"예외 파일 : {len(exception_files)}개")
    uuid_to_file_map = {}
    # UUID 매핑
    for file_list in [tiff_files, other_image_files]:
        for file_path in file_list:
            uuid_str = generate_uuid_str()
            uuid_to_file_map[uuid_str] = file_path

    num_processes = min(cpu_count(), process_count)
    # TIFF 파일 처리
    if tif_mode:
        logging.info(f"TIFF 파일 {len(tiff_files)}개 처리 시작")
        tiff_processed_files = []
        
        # TIFF 파일들을 페이지별로 처리
        for tiff_file in tiff_files:
            try:
                for frame in process_tiff_file(tiff_file, json_data, shared_counters):
                    img_byte_arr = io.BytesIO()
                    frame.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    tiff_processed_files.append((tiff_file, img_byte_arr.getvalue()))
            except Exception as e:
                logging.error(f"TIFF 처리 중 오류 발생: {tiff_file}, 에러: {str(e)}")
                save_as_json(tiff_file, generate_uuid_str(), {}, json_data, success=False)
                shared_counters["tiff_failure"] += 1

        if tiff_processed_files:  # 처리할 페이지가 있는 경우에만 실행
            # 배치 크기 계산
            batch_size = math.ceil(len(tiff_processed_files) / (num_processes * 2))
            batches = []
            
            # 배치 작업 준비
            for i in range(0, len(tiff_processed_files), batch_size):
                batch = tiff_processed_files[i:i + batch_size]
                batches.append((batch, json_data, ocr_server_ip, ocr_server_port, f"TIFF_batch_{i//batch_size}", uuid_to_file_map))
            
            # 멀티프로세싱 실행
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_image_batch, batches)
                
            # TIFF 처리 결과 집계 수정
            for batch_result in results:
                if batch_result:  # 결과가 있는 경우에만 처리
                    for result in batch_result:
                        if result:  # 성공
                            tiff_success += 1
                        else:  # 실패
                            tiff_failure += 1
    elif not tif_mode:
        logging.info(f"TIFF 모드가 비활성화 되어 있어 {len(tiff_files)}개의 TIFF 파일은 처리하지 않습니다.")
        # TIFF 파일들에 대해 처리 실패로 JSON 생성
        for file_path in tiff_files:
            new_uuid = uuid.uuid4()
            utc_now = datetime.datetime.utcnow()
            timestamp = utc_now.timestamp()
            uuid_str = f"{new_uuid}_{timestamp}"
            response = {}
            save_as_json(file_path, uuid_str, response, json_data, success=False)
            tiff_failure += 1
              
    # 일반 이미지 처리
    if other_image_files:
        logging.info(f"일반 이미지 {len(other_image_files)}개 처리 시작")
        filtered_files, _ = filter_large_images(other_image_files, json_data)
        
        # 배치 크기 계산
        batch_size = math.ceil(len(filtered_files) / (num_processes * 2))
        batches = []
        
        # 배치 작업 준비
        for i in range(0, len(filtered_files), batch_size):
            batch = filtered_files[i:i + batch_size]
            batches.append((batch, json_data, ocr_server_ip, ocr_server_port, f"IMG_batch_{i//batch_size}", uuid_to_file_map))
        
        # 멀티프로세싱 실행
        with Pool(processes=num_processes) as pool:
            logging.info(f"멀티프로세싱 실행")
            results = pool.map(process_image_batch, batches)
            
        
    # 전체 결과 출력
    total_success = tiff_success + img_success
    total_failure = tiff_failure + img_failure

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)    

    files_count = len(image_files)

    logging.info(f"총 이미지 파일 개수 : {files_count}")
    logging.info(f"총 분석 시간 : {hours}시간, {minutes}분, {seconds:.2f}초")
    logging.info(f"분석 성공 파일 : {total_success}개, 분석 실패 파일 : {total_failure}")


if __name__ == "__main__":
    main()