from logging.handlers import RotatingFileHandler
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
from PIL import Image
import asyncio
from langdetect import detect
import pytesseract
from multiprocessing import Pool, cpu_count

success_count = 0
failure_count = 0
json_failed_count = 0
current_time = time.strftime("%Y%m%d%H%M%S")
ocr_ver = 1.0
Image.MAX_TEXT_MEMORY = 256 * 1024 * 1024

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
    logging.getLogger("PIL").setLevel(logging.WARNING)

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

file_uuid_map = {}
uuid_file_map = {}

def create_file_uuid_mapping(image_files):
    """
    이미지 파일과 UUID를 매핑하는 함수
    """
    for file_path in image_files:
        new_uuid = uuid.uuid4()
        utc_now = datetime.datetime.utcnow()
        timestamp = utc_now.timestamp()
        uuid_str = f"{new_uuid}_{timestamp}"
        file_uuid_map[file_path] = uuid_str
        uuid_file_map[uuid_str] = file_path
    return file_uuid_map, uuid_file_map

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
            
            language_mapping = {
                'ko': 'ko',
                'ja': 'ja',
                'zh-cn': 'ch_sim',
                'zh-tw': 'ch_sim',
                'en': 'en'
            }
            
            # 문자 범위 체크를 위한 함수
            def has_korean(text):
                return any(ord('가') <= ord(c) <= ord('힣') for c in text)
                
            def has_japanese(text):
                # 히라가나, 가타카나 범위
                return any((ord('ぁ') <= ord(c) <= ord('ゖ')) or 
                          (ord('ァ') <= ord(c) <= ord('ヺ')) for c in text)
                
            def has_chinese(text):
                # 한자 범위 (간체/번체 모두 포함)
                return any(ord('一') <= ord(c) <= ord('龯') for c in text)
            
            # 문자 종류별 비율 계산
            total_len = len(text)
            eng_ratio = len([c for c in text if c.isascii()]) / total_len
            
            # 문자 종류 확인
            has_kor = has_korean(text)
            has_jpn = has_japanese(text)
            has_chn = has_chinese(text)
            
            # 언어 판별 로직
            if detected in language_mapping:
                return language_mapping[detected]
            elif has_kor:
                return 'ko'
            elif has_jpn:
                return 'ja'
            elif has_chn:
                return 'ch_sim'
            elif eng_ratio > 0.5:
                return 'en'
            else:
                return 'unknown'
                
        except Exception as e:
            logging.debug(f"Language detection failed: {str(e)}, defaulting to 'en'")
            return "unknown"
            
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        return "unknown"


def resize_image_in_memory(width, height, max_pixel_limit, file_path, json_data):
    """
    이미지를 메모리에서 처리할 수 있도록 리사이즈하는 함수
    
    Parameters:
        file_path (str): 이미지 파일 경로
        json_data (dict): 설정 데이터
    
    Returns:
        bytes: 리사이즈된 이미지 데이터
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            image_pixel_size = width * height
            
            # 만약 이미지 픽셀 수가 제한을 초과할 경우 강제로 리사이즈
            if image_pixel_size > max_pixel_limit:
                img.thumbnail((width, height))

            # 메모리에서 처리할 수 있도록 BytesIO로 변환
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format)  # 원본 포맷으로 저장
            img_byte_arr.seek(0)  # 스트림의 처음으로 위치 설정
            
            return img_byte_arr
    except Exception as e:
        # logging.error(f"Error resizing image {file_path}: {str(e)}")
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
        # image = Image.open(io.BytesIO(image_data))
        # grayscale_image = image.convert("L").point(lambda x: x if x > 10 else 0)
        # with io.BytesIO() as img_byte_arr:
        #     grayscale_image.save(img_byte_arr, format=image.format)
        #     return img_byte_arr.getvalue()
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
            grayscale_image.save(output, format='PNG')  # 메모리에만 저장
            return output.getvalue()
    except Exception as e:
        # logging.error(f"Error converting image to grayscale: {str(e)}")
        return image_data


async def send_image_to_server(ocr_server_ip, ocr_server_port, image_files, json_data):
    """
    이미지를 OCR 서버에 전송하는 함수
    """
    url = f"http://{ocr_server_ip}:{ocr_server_port}/ocr/"

    async with aiohttp.ClientSession() as session:
        tasks = []

        for file_path, image_data in image_files:
            uuid_str = file_uuid_map.get(file_path)
            if not uuid_str:
                continue

            if isinstance(image_data, io.BytesIO):
                image_data = image_data.getvalue()

            language = detect_language(image_data)
            if language is None:
                language = "unknown"

            task = asyncio.ensure_future(send_image(session, url, uuid_str, image_data, language))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            try:
                uuid_str, response = await task
                file_path = uuid_file_map.get(uuid_str)
                if file_path:
                    save_as_json(file_path, uuid_str, response, json_data, 
                               success=True if response else False)
            except Exception as e:
                file_path = uuid_file_map.get(uuid_str)
                if file_path:
                    # body_info = {"failure_reason": f"응답 처리 중 오류 발생: {str(e)}"}
                    save_as_json(file_path, uuid_str, {}, json_data, success=False)
                logging.error(f"응답 처리 중 오류 발생: {str(e)}")


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
    global failure_count, success_count
    try:
        # multipart/form-data 형식으로 전송
        form_data = aiohttp.FormData()
        form_data.add_field('file', image_data)
        form_data.add_field('language', language)
        form_data.add_field('uuid_str', uuid_str)
        
        async with session.post(url, data=form_data) as response:
            if response.status == 200:
                return uuid_str, await response.json()
            else:
                return uuid_str, None
    except Exception as e:
        logging.error(f"Error in send_image for UUID {uuid_str}: {str(e)}")
        return uuid_str, None

def process_image(file_path, json_data):
    """
    개별 이미지를 처리하여 필터링하는 함수
    """
    try:
        width, height = get_image_size_from_metadata(file_path)
        max_pixel_limit = json_data['ocr_info']['max_pixel_limit']
        max_width = json_data['ocr_info']['max_width']
        max_height = json_data['ocr_info']['max_height']
        
        if width is None or height is None:
            return None, None
        
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        if width * height > max_pixel_limit or width > max_width or height > max_height:
            image_data = process_image_to_grayscale(image_data)
            image_data = resize_image_in_memory(max_width, max_width, max_pixel_limit, file_path, json_data)
            if image_data is None:
                return None, None  # 리사이즈 실패
            return file_path, image_data  # 리사이즈 성공
        else:
                return file_path, image_data  # 원본 사용
    except Exception as e:
        # logging.error(f"이미지 처리 실패 {file_path}: {str(e)}")
        return None, None

def filter_large_images_parallel(image_files, json_data):
    """
    멀티프로세싱을 활용한 대용량 이미지 필터링 함수
    """
    num_processes = min(cpu_count(), json_data['ocr_info']['process_count'])
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_image, [(file_path, json_data) for file_path in image_files])

    # 결과 분리
    filtered_files = []
    failed_files = []
    for file_path, image_data in results:
        if file_path and image_data:
            filtered_files.append((file_path, image_data))
        elif file_path:  # file_path가 있지만 image_data가 None인 경우
            failed_files.append(file_path)
            # JSON은 이미 process_image 함수에서 생성됨

    return filtered_files, failed_files

def send_images_parallel(ocr_server_ip, ocr_server_port, batch_files, json_data):
    """
    서버 전송을 멀티프로세싱으로 처리하는 함수
    """
    num_processes = min(cpu_count(), json_data['ocr_info']['process_count'])
    with Pool(processes=num_processes) as pool:
        pool.starmap(send_image_batch, [(ocr_server_ip, ocr_server_port, batch, json_data) for batch in batch_files])


def send_image_batch(ocr_server_ip, ocr_server_port, batch, json_data):
    """
    개별 배치를 서버에 전송하는 함수
    """
    asyncio.run(send_image_to_server(ocr_server_ip, ocr_server_port, batch, json_data))


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
    global failure_count
    json_data = config_reading('config.json')

    try:
        if json_data is not None:
            root_path = json_data['root_path']
            datainfopath = json_data['datainfopath']
            source_path = datainfopath['target_path']

            source_path = os.path.join(root_path, source_path)

            datafilter = json_data['datafilter']
            image_extensions = datafilter['image_extensions']

            ocr_info = json_data['ocr_info']
            batch_size = ocr_info["batch_size"]
            ocr_server_port = ocr_info["ocr_server_port"]

            log_to_console = ocr_info["log_to_console"]
            log_to_console = ocr_info["log_to_console"]
            log_to_level = ocr_info["log_to_level"]
            log_to_file = ocr_info["log_to_file"]

            current_directory = os.getcwd()
            current_time = time.strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(current_directory, f"ocr_processor_{current_time}.log")

            setup_logging(log_to_console, log_to_file, log_file_path, log_to_level)

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
    for root, _, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)

    # UUID 매핑 생성
    create_file_uuid_mapping(image_files)

    # 멀티프로세싱으로 대용량 이미지 필터링
    filtered_files, big_size_files = filter_large_images_parallel(image_files, json_data)
    
    # 배치 크기로 파일들을 나누기
    batch_files = [
        filtered_files[i:i + batch_size]
        for i in range(0, len(filtered_files), batch_size)
    ]

    # 멀티프로세싱으로 서버 전송
    send_images_parallel(ocr_server_ip, ocr_server_port, batch_files, json_data)

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)    

    files_count = len(image_files)

    logging.info(f"총 이미지 파일 개수 : {files_count}")
    logging.info(f"총 분석 시간 : {hours}시간, {minutes}분, {seconds:.2f}초")
    logging.info(f"분석 성공 파일 : {success_count}개, 분석 실패 파일 : {failure_count}")



if __name__ == "__main__":
    main()