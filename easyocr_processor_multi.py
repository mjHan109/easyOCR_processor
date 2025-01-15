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
    return f"{new_uuid}_{timestamp}"

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

# def resize_image_in_memory(file_path, json_data):
#     """
#     이미지를 메모리에서 처리할 수 있도록 리사이즈하는 함수
    
#     Parameters:
#         file_path (str): 이미지 파일 경로
#         json_data (dict): 설정 데이터
    
#     Returns:
#         bytes: 리사이즈된 이미지 데이터
#     """
#     max_width = json_data['ocr_info']['max_width']
#     max_height = json_data['ocr_info']['max_height']
    
#     try:
#         # PIL 대신 cv2 사용
#         import cv2
#         import numpy as np
        
#         img = cv2.imread(file_path)
#         if img is None:
#             return None
            
#         height, width = img.shape[:2]
#         if width > max_width or height > max_height:
#             scale = min(max_width/width, max_height/height)
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
#         # Convert to bytes
#         success, encoded_img = cv2.imencode('.png', img)
#         if success:
#             return encoded_img.tobytes()
#         return None
#     except Exception as e:
#         logging.error(f"Error resizing image {file_path}: {str(e)}")
#         return None

def resize_image_in_memory(file_path, json_data):
    """이미지를 메모리에서 처리할 수 있도록 리사이즈하는 함수"""
    max_width = json_data['ocr_info']['max_width']
    max_height = json_data['ocr_info']['max_height']
    
    try:
        
        # GIF 파일인 경우 특별 처리
        if file_path.lower().endswith('.gif'):
            img = convert_gif_to_png(file_path)
            if img is None:
                return None
        else:
            # 이미지 읽기
            img = cv2.imread(file_path)
            if img is None:
                # logging.error(f"Failed to read image: {file_path}")
                return None
            
        # 이미지 크기 확인
        height, width = img.shape[:2]
        if width <= 0 or height <= 0:
            # logging.error(f"Invalid image dimensions for {file_path}: {width}x{height}")
            return None
            
        if width > max_width or height > max_height:
            # 스케일 계산 시 0으로 나누기 방지
            scale = min(
                max_width/max(width, 1),
                max_height/max(height, 1)
            )
            
            new_width = max(int(width * scale), 1)
            new_height = max(int(height * scale), 1)
            
            try:
                img = cv2.resize(img, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                # logging.error(f"OpenCV resize error for {file_path}: {str(e)}")
                return None
                
        # 이미지를 바이트로 변환
        try:
            success, encoded_img = cv2.imencode('.png', img)
            if success:
                return encoded_img.tobytes()
            else:
                logging.error(f"Failed to encode image {file_path}")
                return None
        except Exception as e:
            logging.error(f"Error encoding image {file_path}: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Error resizing image {file_path}: {str(e)}")
        return None

def is_image_too_large(image_bytes, file_path, json_data):
    """
    이미지가 너무 큰지 확인하는 함수

    Parameters:
        image_bytes (bytes): 이미지 데이터
        file_path (str): 이미지 파일 경로
        json_data (dict): 설정 데이터
    
    Returns:
        bool: 이미지가 너무 크면 True, 아니면 False
    """
    ocr_info = json_data['ocr_info']
    MAX_WIDTH = ocr_info['max_width']
    MAX_HEIGHT = ocr_info['max_height']
    max_pixel_limit = ocr_info['max_pixel_limit']

    try:
        width, height = get_image_size_from_metadata(file_path)
        
        # 만약 크기를 추출할 수 없는 경우 처리
        if width is None or height is None:
            return True
        else:
            # 가로, 세로 크기 또는 픽셀 제한을 넘는지 확인
            if width > MAX_WIDTH and height > MAX_HEIGHT:
                image_pixel_size = width * height
                if image_pixel_size > max_pixel_limit:
                    return True
            return False  # 이미지 크기가 제한을 초과하지 않으면 False
    except Exception as e:
        # logging.error(f"Error checking image size for {file_path}: {str(e)}")
        return True

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

# def check_memory_and_resize(file_path, max_pixels, json_data):
#     """
#     메모리 및 이미지 크기 체크 후 리사이즈하는 함수

#     Parameters:
#         file_path (str): 이미지 파일 경로
#         max_pixels (int): 최대 픽셀 수
#         max_width (int): 최대 가로 크기
#         max_height (int): 최대 세로 크기
#         json_data (dict): 설정 데이터
    
#     Returns:
#         bytes: 리사이즈된 이미지 데이터
#     """
#     warnings.filterwarnings('ignore', category=UserWarning)

#     try:
#         with open(file_path, 'rb') as f:
#             image_data = f.read()

#         # 팔레트 이미지 처리 추가
#         if is_image_too_large(file_path, max_pixels, json_data):
#             with Image.open(io.BytesIO(image_data)) as img:
#                 # 팔레트 이미지를 RGB로 변환
#                 if img.mode == 'P':
#                     img = img.convert('RGB')
#                 # RGBA 이미지를 RGB로 변환
#                 elif img.mode == 'RGBA':
#                     background = Image.new('RGB', img.size, (255, 255, 255))
#                     background.paste(img, mask=img.split()[3])
#                     img = background
#                     image_data = process_image_to_grayscale(image_data)

#             image_data = resize_image_in_memory(file_path, json_data)
#             if image_data is None:
#                 body_info={}
#                 new_uuid = uuid.uuid4()
#                 utc_now = datetime.datetime.utcnow()
#                 timestamp = utc_now.timestamp()
#                 uuid_str = str(new_uuid) + '_' + str(timestamp)
#                 save_as_json(file_path, uuid_str, body_info, json_data, success=False)
#                 return None
#         return image_data
#     except Exception as e:
#         # logging.error(f"Error processing image {file_path}: {str(e)}")
#         return None

def check_memory_and_resize(file_path, max_pixels, json_data):
    """
    이미지 리사이즈만 처리하는 함수 (크기 체크는 이미 완료)
    """
    try:
        # PIL 대신 cv2 사용
        img = cv2.imread(file_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            return None

        height, width = img.shape[:2]
        if width * height > max_pixels:
            ratio = math.sqrt(max_pixels / (width * height))
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 메모리 사용량을 줄이기 위해 이미지 압축
        encode_params = [
            cv2.IMWRITE_PNG_COMPRESSION, 9,
            cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT,
            cv2.IMWRITE_PNG_BILEVEL, 0
        ]
        
        success, encoded_img = cv2.imencode('.png', img, encode_params)
        if not success:
            return None
            
        # 메모리 정리
        img = None
        return encoded_img.tobytes()

    except Exception as e:
        logging.error(f"이미지 처리 중 오류 발생 {file_path}: {str(e)}")
        return None

def filter_large_images(image_files, json_data):
    """
    대용량 이미지 필터링 함수

    Parameters:
        image_files (list): 이미지 파일 경로 리스트
        json_data (dict): 설정 데이터
    
    Returns:
        tuple: 필터링된 이미지 파일 리스트와 대용량 이미지 파일 리스트
    """
    filtered_files = []
    big_size_files = []
    
    max_pixel_limit = json_data['ocr_info']['max_pixel_limit']

    for file_path in image_files:
        try:
            # EMF/WMF 파일 건너뛰기
            if file_path.lower().endswith(('.emf', '.wmf')):
                new_uuid = uuid.uuid4()
                utc_now = datetime.datetime.utcnow()
                timestamp = utc_now.timestamp()
                uuid_str = f"{new_uuid}_{timestamp}"
                body_info = {}
                save_as_json(file_path, uuid_str, body_info, json_data, success=False)
                continue

            image_data = check_memory_and_resize(file_path, max_pixel_limit, json_data)

            if image_data is None:
                continue
            else:
                width, height = get_image_size_from_metadata(file_path)
                new_uuid = uuid.uuid4()
                utc_now = datetime.datetime.utcnow()
                timestamp = utc_now.timestamp()
                uuid_str = f"{new_uuid}_{timestamp}"
                if width is not None and height is not None:
                    if (width * height) > max_pixel_limit:
                        body_info = {} 
                        save_as_json(file_path, uuid_str, body_info, json_data, success=False)
                        big_size_files.append(file_path)
                    else:
                        filtered_files.append((file_path, image_data))
                else:
                    body_info = {} 
                    save_as_json(file_path, uuid_str, body_info, json_data, success=False)
        except Exception as e:
            new_uuid = uuid.uuid4()
            utc_now = datetime.datetime.utcnow()
            timestamp = utc_now.timestamp()
            uuid_str = f"{new_uuid}_{timestamp}"
            logging.error(f"Error reading file {file_path}: {str(e)}")
            body_info = {} 
            save_as_json(file_path, uuid_str, body_info, json_data, success=False)

    return filtered_files, big_size_files

uuid_to_file_map = {}

async def send_image_to_server(ocr_server_ip, ocr_server_port, image_files, json_data):
    """
    이미지를 OCR 서버에 전송하는 함수

    Parameters:
        ocr_server_ip (str): OCR 서버 IP 주소
        ocr_server_port (int): OCR 서버 포트
        image_files (list): 이미지 파일 리스트
        json_data (dict): 설정 데이터
    """

    url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"

    async with aiohttp.ClientSession() as session:
        tasks = []

        # image_files는 이미 배치 크기로 나눠진 리스트를 받음
        for file_path, image_data in image_files:
            # UUID 생성 및 파일명 매핑
            new_uuid = uuid.uuid4()
            utc_now = datetime.datetime.utcnow()
            timestamp = utc_now.timestamp()
            uuid_str = f"{new_uuid}_{timestamp}"
            uuid_to_file_map[uuid_str] = file_path

            # BytesIO를 bytes로 변환
            if isinstance(image_data, io.BytesIO):
                image_data = image_data.getvalue()

            # 언어 감지 후 서버에 전송할 언어 정보와 함께 전송
            language = detect_language(image_data)
            if language is None:
                # logging.warning(f"{file_path}의 언어를 감지 할 수 없어 기본 언어로 설정합니다.")
                language = "unknown"

            success = False

            task = asyncio.ensure_future(send_image(session, url, uuid_str, image_data, language))
            tasks.append(task)

        # 응답 처리
        for task in asyncio.as_completed(tasks):
            try:
                uuid_str, response, success = await task
                file_path = uuid_to_file_map.get(uuid_str)
                if file_path:
                    # JSON 저장
                    save_as_json(file_path, uuid_str, response, json_data, success)
                        
            except Exception as e:
                logging.error(f"응답 처리 중 오류 발생: {str(e)}")

        # 배치 완료 후 남은 리소스 정리
        uuid_to_file_map.clear()
        tasks.clear()


async def send_image(session, url, uuid_str, image_data, language):
    """
    이미지를 OCR 서버에 전송하는 함수

    Parameters:
        session (aiohttp.ClientSession): 클라이언트 세션
        url (str): OCR 서버 URL
        uuid_str (str): UUID 문자열
        image_data (bytes): 이미지 데이터
        language (str): 언어 코드
    """
    try:
        # multipart/form-data 형식으로 전송
        form_data = aiohttp.FormData()
        form_data.add_field('file', image_data)
        form_data.add_field('language', language)
        form_data.add_field('uuid_str', uuid_str)
        
        async with session.post(url, data=form_data) as response:
            if response.status == 200:
                result = await response.json()
                return uuid_str, result, True
            else:
                return uuid_str, None, False
    except Exception as e:
        logging.error(f"Error in send_image for UUID {uuid_str}: {str(e)}")
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


def process_tiff_file(file_path, json_data):
    """
    TIFF 파일의 각 페이지를 개별적으로 처리하는 함수
    
    Parameters:
        file_path (str): TIFF 파일 경로
        json_data (dict): 설정 데이터
    
    Returns:
        list: (파일경로, 이미지데이터) 튜플의 리스트
    """
    processed_pages = []
    try:
        with Image.open(file_path) as img:
            n_frames = getattr(img, 'n_frames', 1)  # TIFF 파일의 총 페이지 수

            for i in range(n_frames):
                try:
                    img.seek(i)
                    # TIFF 파일 모드 처리
                    if img.mode == 'P':
                        frame = img.convert('RGB')
                    elif img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        frame = background
                    elif img.mode not in ['L', 'RGB']:
                        frame = img.convert('RGB')
                    else:
                        frame = img.copy()

                    # 이미지 리사이즈
                    frame = resize_image_tiff(frame, json_data)

                    # 메모리에 이미지 저장
                    img_byte_arr = io.BytesIO()
                    frame.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    # 파일 경로에 페이지 번호 추가
                    page_file_path = f"{file_path}_page_{i+1}" if n_frames > 1 else file_path
                    processed_pages.append((page_file_path, img_byte_arr.getvalue()))
                      
                except Exception as e:
                    logging.error(f"Error processing TIFF page {i} in {file_path}: {str(e)}")
                    continue
                    
    except Exception as e:
        logging.error(f"Error opening TIFF file {file_path}: {str(e)}")
        return []

    return processed_pages

async def send_tiff_pages_to_server(session, ocr_server_ip, ocr_server_port, uuid_str, pages_data, language):
    """
    TIFF 파일의 모든 페이지를 서버에 전송하고 결과를 합치는 함수
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
            with open(json_file_name, 'w', encoding='utf-8') as json_file:
                json_file.write(json_string)
        except Exception as e:
            json_failed_count +=1
            error_message = f"JSON 파일 생성 중 알 수 없는 오류 발생 : {str(e)}"
            logging.error(error_message)

    return json_failed_count

async def process_batch(batch_files, json_data, ocr_server_ip, ocr_server_port, shared_counters):
    """
    배치 단위로 이미지를 처리하는 함수
    """
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            uuid_to_file_map = {}

            for file_path in batch_files:
                if isinstance(file_path, tuple):
                    file_path, image_data = file_path
                else:
                    image_data = None

                new_uuid = uuid.uuid4()
                utc_now = datetime.datetime.utcnow()
                timestamp = utc_now.timestamp()
                uuid_str = f"{new_uuid}_{timestamp}"
                uuid_to_file_map[uuid_str] = file_path

                if image_data is None:
                    with open(file_path, 'rb') as f:
                        image_data = f.read()

                language = detect_language(image_data)
                if language is None:
                    language = "unknown"

                task = asyncio.ensure_future(
                    send_image(session, f"http://{ocr_server_ip}:{ocr_server_port}/ocr/",
                             uuid_str, image_data, language)
                )
                tasks.append(task)

            for task in asyncio.as_completed(tasks):
                try:
                    uuid_str, response = await task
                    file_path = uuid_to_file_map.get(uuid_str)
                    if file_path:
                        success = True if response else False
                        save_as_json(file_path, uuid_str, response, json_data, success)
                        
                        # 공유 카운터 업데이트
                        if success:
                            shared_counters['success_count'] += 1
                        else:
                            shared_counters['failure_count'] += 1
                            
                except Exception as e:
                    logging.error(f"응답 처리 중 오류 발생: {str(e)}")
                    shared_counters['failure_count'] += 1

    except Exception as e:
        logging.error(f"배치 처리 중 오류 발생: {str(e)}")

def process_image_batch(args):
    """
    각 프로세스에서 실행될 이미지 배치 처리 함수
    """
    batch_files, json_data, ocr_server_ip, ocr_server_port, batch_id = args
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(process_batch_async(batch_files, json_data, 
                                                            ocr_server_ip, ocr_server_port, 
                                                            batch_id))
        loop.close()
        return results
        
    except Exception as e:
        logging.error(f"배치 {batch_id} 처리 중 오류 발생: {str(e)}")
        return []

async def prepare_image(file_path, session, url):
    """이미지 준비 작업을 비동기로 처리"""
    try:
        # 파일 읽기를 비동기로 처리
        async with aiofiles.open(file_path, 'rb') as f:
            image_data = await f.read()
        
        uuid_str = generate_uuid_str()
        
        # 언어 감지는 ThreadPoolExecutor를 사용하여 비동기로 처리
        loop = asyncio.get_event_loop()
        language = await loop.run_in_executor(None, detect_language, image_data)
        
        return uuid_str, file_path, image_data, language
    except Exception as e:
        logging.error(f"Error preparing {file_path}: {str(e)}")
        return None

# async def process_batch_async(batch_files, json_data, ocr_server_ip, ocr_server_port, batch_id):
    """
    배치 내 이미지들을 비동기로 처리
    """
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        file_map = {}  # UUID to file_path mapping

        for file_path in batch_files:
            if isinstance(file_path, tuple):
                file_path, image_data = file_path
            else:
                #  HEIC 파일 처리
                if file_path.lower().endswith(('.heic', '.heif')):
                    logging.info(f"HEIC 파일 처리 : {file_path}")
                    try:
                        image_data = convert_heic_to_png(file_path)
                        if image_data is None:
                            # HEIC 변환 실패 시
                            new_uuid = generate_uuid_str()
                            save_as_json(file_path, new_uuid, {}, json_data, success=False)
                            continue
                    except Exception as e:
                        # logging.error(f"HEIC conversion failed for {file_path}: {str(e)}")
                        new_uuid = generate_uuid_str()
                        save_as_json(file_path, new_uuid, {}, json_data, success=False)
                        continue
                else:
                    try:
                        with open(file_path, 'rb') as f:
                            image_data = f.read()
                    except Exception as e:
                        # logging.error(f"파일 읽기 실패 (배치 {batch_id}): {file_path}, 오류: {str(e)}")
                        continue

            uuid_str = f"{uuid.uuid4()}_{time.time()}"
            file_map[uuid_str] = file_path

            language = detect_language(image_data)
            if language is None:
                language = "unknown"

            task = send_image(session, f"http://{ocr_server_ip}:{ocr_server_port}/ocr/",
                            uuid_str, image_data, language)
            tasks.append((uuid_str, task))

        for uuid_str, task in tasks:
            try:
                _, response_data, success = await task
                file_path = file_map[uuid_str]
                
                # JSON 파일 저장
                save_as_json(file_path, uuid_str, response_data, json_data, success)
                results.append((file_path))
                
            except Exception as e:
                logging.error(f"응답 처리 실패 (배치 {batch_id}): {str(e)}")
                results.append((file_map[uuid_str], False))

    return results

async def process_batch_async(batch_files, json_data, ocr_server_ip, ocr_server_port, batch_id):
    async with aiohttp.ClientSession() as session:
        url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"
        
        # 이미지 준비 작업을 동시에 실행
        prepare_tasks = [prepare_image(f, session, url) for f in batch_files]
        prepared_images = await asyncio.gather(*prepare_tasks)
        
        # 유효한 결과만 필터링
        valid_images = [img for img in prepared_images if img is not None]
        
        # OCR 요청을 동시에 전송
        ocr_tasks = []
        for uuid_str, file_path, image_data, language in valid_images:
            task = send_image(session, url, uuid_str, image_data, language)
            ocr_tasks.append((uuid_str, file_path, task))
        
        # 결과 처리
        results = []
        for uuid_str, file_path, task in ocr_tasks:
            try:
                response = await task
                results.append((file_path, response))
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                results.append((file_path, None))
        
        return results

def main():
    """
    메인 함수
    """ 
    global tiff_success, tiff_failure, img_success, img_failure
    json_data = config_reading('config.json')

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
    bmp_files = []

    # 파일 분류
    tiff_files, other_image_files = [], []
    for root, _, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(file_path)
            elif file_path.lower().endswith(('.bmp', ".emf", ".wmf")):
                new_uuid = generate_uuid_str()
                save_as_json(file_path, new_uuid, {}, json_data, success=False)
                img_failure += 1
            elif any(file_path.lower().endswith(ext) for ext in image_extensions):
                other_image_files.append(file_path)

    num_processes = min(cpu_count(), process_count)
    
    # TIFF 파일 처리
    if tif_mode and tiff_files:
        logging.info(f"TIFF 파일 {len(tiff_files)}개 처리 시작")
        tiff_processed_files = []
        
        # TIFF 파일들을 페이지별로 처리
        for tiff_file in tiff_files:
            try:
                processed_pages = process_tiff_file(tiff_file, json_data)
                if processed_pages:  # None이 아닐 경우에만 추가
                    tiff_processed_files.extend(processed_pages)
                else:
                    # 처리 실패한 TIFF 파일에 대한 JSON 생성
                    new_uuid = uuid.uuid4()
                    utc_now = datetime.datetime.utcnow()
                    timestamp = utc_now.timestamp()
                    uuid_str = f"{new_uuid}_{timestamp}"
                    response = {}
                    save_as_json(tiff_file, uuid_str, response, json_data, success=False)
                    tiff_failure += 1
            except Exception as e:
                logging.error(f"TIFF 파일 처리 중 오류 발생: {tiff_file}, {str(e)}")
                new_uuid = uuid.uuid4()
                utc_now = datetime.datetime.utcnow()
                timestamp = utc_now.timestamp()
                uuid_str = f"{new_uuid}_{timestamp}"
                response = {}
                save_as_json(tiff_file, uuid_str, response, json_data, success=False)
                tiff_failure += 1
        
        if tiff_processed_files:  # 처리할 페이지가 있는 경우에만 실행
            # 배치 크기 계산
            batch_size = math.ceil(len(tiff_processed_files) / (num_processes * 2))
            batches = []
            
            # 배치 작업 준비
            for i in range(0, len(tiff_processed_files), batch_size):
                batch = tiff_processed_files[i:i + batch_size]
                batches.append((batch, json_data, ocr_server_ip, ocr_server_port, f"TIFF_batch_{i//batch_size}"))
            
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
    
    elif not tif_mode and tiff_files:
        logging.info(f"TIFF 모드가 비활성화되어 있어 {len(tiff_files)}개의 TIFF 파일은 처리하지 않습니다.")
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
            batches.append((batch, json_data, ocr_server_ip, ocr_server_port, f"IMG_batch_{i//batch_size}"))
        
        # 멀티프로세싱 실행
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_image_batch, batches)
            
        # 일반 이미지 처리 결과 집계 (수정된 부분)
        img_success = len([result for batch_result in results for result in batch_result if result])
        img_failure = len([result for batch_result in results for result in batch_result if not result])

        
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