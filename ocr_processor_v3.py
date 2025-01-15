import sys
from tkinter import Image
import aiohttp
import asyncio
import os
import logging
import json
import uuid
from typing import Dict, List
import time
from datetime import datetime
import pytesseract
from PIL import Image
from charset_normalizer import detect
import aiofiles
import io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent.futures
import numpy as np
import cv2
from functools import lru_cache
import warnings

# PIL의 이미지 처리 관련 경고 억제
from PIL import Image
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# 환경 변수 설정으로 경고 메시지 억제
os.environ['PYTHONWARNINGS'] = 'ignore'

class OCRClient:
    def __init__(self, server_url: str, config_path: str):
        self.server_url = server_url
        self.config = self.load_config(config_path)
        
        # config에서 worker 설정 가져오기
        max_workers_config = self.config['ocr_info']['max_workers']
        self.max_processes = max_workers_config['processes'] or os.cpu_count()
        self.max_threads = max_workers_config['threads']
        
        self.session = None
        self.semaphore = asyncio.Semaphore(self.max_threads)
        self.file_uuid_map = {}
        self.uuid_file_map = {}
        
    @lru_cache(maxsize=10000)
    def create_uuid_mapping(self, file_path: str) -> str:
        """파일 경로에 대한 UUID 생성 및 매핑 (캐시 적용)"""
        if file_path in self.file_uuid_map:
            return self.file_uuid_map[file_path]
            
        new_uuid = f"{uuid.uuid4()}_{int(time.time())}"  
        self.file_uuid_map[file_path] = new_uuid
        self.uuid_file_map[new_uuid] = file_path
        return new_uuid

    def create_file_uuid_mappings(self, directory: str) -> None:
        """디렉토리 내 모든 이미지 파일에 대한 UUID 매핑 생성"""
        try:
            logging.info(f"UUID 매핑 생성 시작: {directory}")
            supported_formats = set(self.config['datafilter']['image_extensions'])
            file_count = 0
            
            # 모든 이미지 파일에 대해 UUID 매핑 생성
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_formats:
                        file_path = os.path.join(root, filename)
                        if os.path.exists(file_path):
                            self.create_uuid_mapping(file_path)
                            file_count += 1
            
            logging.info(f"UUID 매핑 완료: 총 {file_count}개 이미지 파일 처리됨")
            
        except Exception as e:
            logging.error(f"UUID 매핑 생성 중 오류 발생: {str(e)}")

    def clear_uuid_cache(self):
        """UUID 캐시 초기화"""
        self.create_uuid_mapping.cache_clear()
        self.file_uuid_map.clear()
        self.uuid_file_map.clear()

    def get_uuid_mapping(self, file_path: str) -> str:
        """파일 경로에 대한 UUID 조회"""
        return self.file_uuid_map.get(file_path)

    def get_file_path(self, uuid: str) -> str:
        """UUID에 대한 파일 경로 조회"""
        return self.uuid_file_map.get(uuid)

    # async def process_directory(self, directory: str) -> List[Dict]:
    #     """디렉토리 내 모든 이미지를 병렬 처리"""
    #     self.create_file_uuid_mappings(directory)
    #     files = list(self.file_uuid_map.keys())
    #     total_files = len(files)
        
    #     logging.info(f"총 처리할 파일 수: {total_files}")
        
    #     # 파일들을 배치로 나누기
    #     batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
    #     results = []
    #     processed = 0
        
    #     async with aiohttp.ClientSession() as session:
    #         self.session = session
            
    #         # 배치 단위로 병렬 처리
    #         for batch in batches:
    #             # 배치 내 파일들을 동시에 처리
    #             batch_tasks = []
    #             for file_path in batch:
    #                 task = asyncio.create_task(self._process_with_semaphore(file_path))
    #                 batch_tasks.append(task)
                
    #             # 배치 결과 수집
    #             batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    #             valid_results = [r for r in batch_results if isinstance(r, dict)]
    #             results.extend(valid_results)
                
    #             processed += len(batch)
    #             logging.info(f"진행률: {processed}/{total_files} ({processed/total_files*100:.1f}%)")
                
    #     return results

    async def _process_with_semaphore(self, file_path: str) -> Dict:
        """세마포어를 사용한 단일 파일 처리"""
        async with self.semaphore:
            return await self.process_single_image(file_path, None)

    # async def process_single_image(self, file_path: str, language: str = None) -> Dict:
    #     """단일 이미지 처리 최적화"""
    #     try:
    #         # 기본 검사
    #         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    #             return self._create_error_response(file_path, "파일이 없거나 비어있음")

    #         # 파일 확장자 체크
    #         ext = os.path.splitext(file_path)[1].lower()
    #         if ext in ['.tif', '.tiff']:
    #             return await self._process_tiff(file_path)
    #         elif ext == '.emf':
    #             return self._create_error_response(file_path, "EMF 파일은 지원하지 않음")

    #         # 일반 이미지 처리
    #         try:
    #             with open(file_path, 'rb') as f:
    #                 image_data = f.read()

    #             uuid_str = self.create_uuid_mapping(file_path)
    #             form = aiohttp.FormData()
    #             form.add_field('file', image_data)
    #             form.add_field('language', language or 'auto')
    #             form.add_field('uuid_str', uuid_str)

    #             async with self.session.post(f"{self.server_url}/ocr/", data=form) as response:
    #                 if response.status != 200:
    #                     return self._create_error_response(file_path, f"서버 오류: {response.status}")
                    
    #                 response_data = await response.json()
    #                 return {
    #                     "file_path": file_path,
    #                     "uuid": uuid_str,
    #                     "success": True,
    #                     "text": response_data.get("text", ""),
    #                     "language": language or "auto"
    #                 }

    #         except Exception as e:
    #             return self._create_error_response(file_path, f"처리 오류: {str(e)}")

    #     except Exception as e:
    #         return self._create_error_response(file_path, f"예상치 못한 오류: {str(e)}")

    def _create_error_response(self, file_path: str, error_msg: str) -> Dict:
        """에러 응답 생성 헬퍼 함수"""
        return {
            "file_path": file_path,
            "uuid": self.create_uuid_mapping(file_path),
            "success": False,
            "error": error_msg
        }

    async def _process_tiff(self, file_path: str) -> Dict:
        """TIFF 파일 처리"""
        try:
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                text = await loop.run_in_executor(executor, process_tiff, file_path)
                
            uuid_str = self.create_uuid_mapping(file_path)
            return {
                "file_path": file_path,
                "uuid": uuid_str,
                "success": bool(text),
                "text": text
            }
        except Exception as e:
            return self._create_error_response(file_path, f"TIFF 처리 오류: {str(e)}")

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ocr_client.log')
            ]
        )

    @staticmethod
    def get_main_fold(target_path: str, file_path: str) -> str:
        """메인 폴더 경로 추출"""
        try:
            relative_path = os.path.relpath(file_path, target_path)
            main_folder = os.path.dirname(relative_path).split(os.sep)[0]
            return os.path.join(target_path, main_folder)
        except Exception as e:
            logging.error(f"메인 폴더 경로 추출 실패: {str(e)}")
            return ""

    @staticmethod
    def read_file_from_path(file_path: str) -> Dict:
        """파일 메타 정보 읽기"""
        try:
            file_stats = os.stat(file_path)
            return {
                "accessed": time.ctime(file_stats.st_atime),
                "created": time.ctime(file_stats.st_ctime),
                "mtime": time.ctime(file_stats.st_mtime),
                "owner": f"{file_stats.st_uid}:{file_stats.st_gid}"
            }
        except Exception as e:
            logging.error(f"파일 메타 정보 읽기 실패: {str(e)}")
            return {}
        
    def process_batch_worker(self, batch_data):
        """배치 처리를 위한 워커 함수"""
        batch, client = batch_data
        try:
            # 배치별 언어 감지
            language = client.detect_language_for_batch(batch)
            
            # 배치 처리
            results = []
            success_count = 0
            
            for file_path in batch:
                result = asyncio.run(client.process_single_image(file_path, language))
                if asyncio.run(client.save_result(result, client.config)):
                    success_count += 1
                results.append(result)
                
            return results, success_count, len(batch)
        except Exception as e:
            logging.error(f"배치 처리 중 오류 발생: {str(e)}")
            return [], 0, len(batch)

    async def process_directory(self, directory: str) -> List[Dict]:
        """디렉토리 내 모든 이미지를 멀티프로세싱으로 처리"""
        self.create_file_uuid_mappings(directory)
        batch_size = self.config['ocr_info']['batch_size']
        total_files = len(self.file_uuid_map)
        processed_files = 0
        success_count = 0
        fail_count = 0
        
        logging.info(f"총 처리할 파일 수: {total_files}")
        
        # 파일 목록을 배치로 나누기
        file_paths = list(self.file_uuid_map.keys())
        batch_size = max(batch_size, total_files // (self.max_processes * 4))
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        
        results = []
        
        # ProcessPoolExecutor 설정
        with ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            # 각 배치에 대한 작업 준비
            batch_data = [(batch, self) for batch in batches]
            
            # 배치 작업 실행
            futures = [executor.submit(self.process_batch_worker, data) for data in batch_data]
            
            # 결과 수집
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results, batch_success, batch_size = future.result()
                    processed_files += batch_size
                    success_count += batch_success
                    fail_count += (batch_size - batch_success)
                    results.extend(batch_results)
                    
                    # logging.info(f"배치 처리 완료: {success_count}/{batch_size} 성공 " +
                    #             f"(전체 진행률: {processed_files}/{total_files}, {(processed_files/total_files*100):.1f}%)")
                    
                except Exception as e:
                    logging.error(f"배치 처리 결과 수집 중 오류: {str(e)}")
                    continue
                
                # 메모리 관리를 위한 짧은 대기
                await asyncio.sleep(0.1)
        
        logging.info(f"전체 처리 완료:")
        logging.info(f"- 총 처리 파일: {len(results)}개")
        logging.info(f"- 성공: {success_count}개")
        logging.info(f"- 실패: {fail_count}개")
        logging.info(f"- 성공률: {(success_count/total_files*100):.1f}%")
        
        return results



    async def process_single_image(self, file_path: str, language: str) -> Dict:
        """단일 이미지 처리"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")

            # 파일 크기 확인
            if os.path.getsize(file_path) == 0:
                logging.error(f"빈 파일: {file_path}")
                return {
                    "file_path": file_path,
                    "uuid": self.create_uuid_mapping(file_path),
                    "success": False
                }

            # TIFF 파일 체크 및 처리
            if file_path.lower().endswith(('.tif', '.tiff')):
                try:
                    uuid_str = self.create_uuid_mapping(file_path)
                    with ProcessPoolExecutor(max_workers=1) as executor:
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(executor, process_tiff, file_path)
                        
                    return {
                        "file_path": file_path,
                        "uuid": uuid_str,
                        "success": True if text else False,
                        "text": text
                    }
                except Exception as e:
                    logging.error(f"TIFF 처리 오류 ({file_path}): {str(e)}")
                    return {
                        "file_path": file_path,
                        "uuid": self.create_uuid_mapping(file_path),
                        "success": False,
                        "error": f"TIFF 처리 오류: {str(e)}"
                    }

            # EMF 파일 체크
            elif file_path.lower().endswith('.emf'):
                return {
                    "file_path": file_path,
                    "uuid": self.create_uuid_mapping(file_path),
                    "success": False,
                    "error": "EMF 파일은 지원하지 않음"
                }

            else:
                try:
                    # 이미지 파일 읽기
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
                    
                    if not image_data:
                        raise ValueError("이미지 데이터가 비어있음")
                    
                    # ProcessPoolExecutor를 사용하여 이미지 전처리
                    with ProcessPoolExecutor(max_workers=1) as executor:
                        loop = asyncio.get_event_loop()
                        processed_image = await loop.run_in_executor(
                            executor, 
                            preprocess_image_worker, 
                            image_data
                        )
                    
                    # UUID 생성 및 form 데이터 준비
                    uuid_str = self.create_uuid_mapping(file_path)
                    form = aiohttp.FormData()
                    form.add_field('file', 
                                 processed_image if processed_image else image_data,
                                 filename=os.path.basename(file_path))
                    form.add_field('language', 
                                 language if language else 'ko')  # 기본값으로 'ko' 설정
                    form.add_field('uuid_str', uuid_str)
                    
                    # 서버 요청
                    async with aiohttp.ClientSession() as session:
                        url = f"{self.server_url}/ocr/"
                        async with session.post(url, data=form) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"서버 응답 오류 (상태 코드: {response.status})")
                            
                            response_text = await response.text()
                            if not response_text:
                                raise Exception("빈 응답 받음")
                            
                            response_data = json.loads(response_text)
                            if not response_data:
                                raise Exception("유효하지 않은 응답 데이터")
                            
                            return {
                                "file_path": file_path,
                                "uuid": uuid_str,
                                "success": True,
                                "text": response_data.get("text", ""),
                                "language": language
                            }
                        
                except Exception as e:
                    logging.error(f"이미지 처리 오류 ({file_path}): {str(e)}")
                    return {
                        "file_path": file_path,
                        "uuid": self.create_uuid_mapping(file_path),
                        "success": False,
                        "error": f"처리 오류: {str(e)}"
                    }

        except Exception as e:
            logging.error(f"예상치 못한 오류 ({file_path}): {str(e)}")
            return {
                "file_path": file_path,
                "uuid": self.create_uuid_mapping(file_path),
                "success": False,
                "error": f"예상치 못한 오류: {str(e)}"
            }


    def process_result(self, result: Dict) -> Dict:
        """서버 응답 처리 및 파일 경로 매핑"""
        try:
            uuid_str = result.get('uuid')
            file_path = self.uuid_file_map.get(uuid_str)
            
            if file_path:
                result['file_path'] = file_path
                # UUID 매핑 정리
                self.uuid_file_map.pop(uuid_str, None)
                self.file_uuid_map.pop(file_path, None)
            
            return result
        except Exception as e:
            logging.error(f"Error processing result: {str(e)}")
            return result

    async def save_result(self, result: Dict, json_data: Dict):
        """OCR 결과를 JSON 파일로 비동기적으로 저장"""
        try:
            file_path = result.get('file_path')
            uuid_str = result.get('uuid')
            
            if not file_path or not uuid_str:
                logging.error("파일 경로 또는 UUID 없음")
                return False
            
            root_path = json_data["root_path"]
            target_path = json_data["datainfopath"]["target_path"]
            target_path = os.path.join(root_path, target_path)
            main_directory = self.get_main_fold(target_path, file_path)
            
            file_name, file_extension = os.path.splitext(os.path.basename(file_path))
            es_target_path = json_data['elasticsearch']['normal_el_file_target_path']
            es_filepath = json_data['elasticsearch']['el_file_path']
            now = datetime.now()
            
            relative_path = os.path.relpath(file_path, root_path)
            root_folder = os.path.dirname(relative_path)
            full_directory = os.path.normpath(os.path.join(root_path, root_folder))
            meta_info = self.read_file_from_path(file_path)
            
            # OCR 데이터 구조 생성
            ocr_data = {
                "json_write_time": now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                "root_path": main_directory,
                "directory": full_directory,
                "uuid": uuid_str,
                "file": {
                    "accessed": meta_info.get("accessed", ""),
                    "ctime": meta_info.get("created", ""),
                    "mtime": meta_info.get("mtime", ""),
                    "owner": meta_info.get("owner", ""),
                    "path": file_path,
                    "mime_type": f"image/{file_extension.lstrip('.')}",
                    "size": os.path.getsize(file_path),
                    "type": "file",
                    "extension": file_extension.lstrip('.')
                },
                "title": f"{file_name}{file_extension}",
                "tags": ["ocr", "file", "S"] if result.get('success', False) else ["ocr", "file", "N", "exception"]
            }

            # success 값에 따른 명확한 분기 처리
            if result['success'] is True and result['text']:  # 성공이고 텍스트가 있는 경우
                ocr_data["tags"] = ["ocr", "file", "S"]
                ocr_data["content"] = result['text']
                ocr_data["summary"] = result['text'][:300]
            else:  # 실패했거나 텍스트가 없는 경우
                ocr_data["tags"] = ["ocr", "file", "N", "exception"]

            # 결과 저장
            result_path = os.path.join(es_target_path, es_filepath)
            os.makedirs(result_path, exist_ok=True)
            
            json_file_name = os.path.join(result_path, f"{uuid_str}.json")
            async with aiofiles.open(json_file_name, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(ocr_data, ensure_ascii=False, indent=4))
                
            return True
            
        except Exception as e:
            logging.error(f"JSON 저장 중 오류 발생: {str(e)}")
            return False
        
    def detect_language_for_directory(self, directory_path: str) -> str:
        """디렉토리 내의 이미지들의 언어를 감지하는 함수"""
        try:
            # logging.info(f"디렉토리 언어 감지 시작: {directory_path}")
            supported_formats = set(self.config['datafilter']['image_extensions'])
            processed_count = 0
            
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in supported_formats:
                        processed_count += 1
                        image_path = os.path.join(root, file)
                        try:
                            # OpenCV로 이미지 읽기
                            img = cv2.imread(image_path)
                            if img is not None:
                                # 그레이스케일로 변환
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                # tesseract에 직접 numpy 배열 전달
                                text = pytesseract.image_to_string(gray, lang='kor+eng+jpn+chi_sim')
                                if text.strip():
                                    detected_lang = detect(text)
                                    lang_map = {
                                        'ko': 'ko',
                                        'ja': 'ja',
                                        'zh': 'zh',
                                        'en': 'en'
                                    }
                                    return lang_map.get(detected_lang, 'ko')
                        except:
                            continue
            
            return 'ko'
            
        except Exception as e:
            return 'ko'
        
    def detect_language_for_batch(self, image_data: bytes) -> str:
        """이미지 데이터의 언어를 감지하는 함수"""
        try:
            # 바이트 데이터를 numpy 배열로 변환
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # 그레이스케일로 변환
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # tesseract에 직접 numpy 배열 전달
                text = pytesseract.image_to_string(gray, lang='kor+eng+jpn+chi_sim')
                if text.strip():
                    detected_lang = detect(text)
                    lang_map = {
                        'ko': 'ko',
                        'ja': 'ja',
                        'zh': 'zh',
                        'en': 'en'
                    }
                    return lang_map.get(detected_lang, 'ko')
            return 'ko'
                    
        except Exception as e:
            return 'ko'

def setup_logging(config):
    """로깅 설정을 초기화하는 함수"""
    # 로깅 설정
    log_level = config['ocr_info']['log_to_level']
    log_to_console = config['ocr_info']['log_to_console'] 
    log_to_file = config['ocr_info']['log_to_file']

    # 로그 레벨 매핑
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO, 
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    # 기본 로그 레벨 설정 
    log_level = level_map.get(log_level.replace('logging.',''), logging.ERROR)
    
    # 로그 핸들러 설정
    handlers = []
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
        
    if log_to_file:
        log_path = config.get('log_path', 'logs')
        os.makedirs(log_path, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(log_path, f'ocr_processor_{current_time}.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # 모든 경고 메시지 억제 설정
    logging.getLogger('PIL').setLevel(logging.CRITICAL)
    logging.getLogger('libpng').setLevel(logging.CRITICAL)
    
    # 추가 경고 메시지 필터링
    for logger_name in ['PIL', 'libpng', 'pytesseract', 'charset_normalizer']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
    # 기존 로깅 설정 유지
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )

# def detect_language(image_path: str) -> str:
#     """
#     이미지의 텍스트 언어를 감하 함수
    
#     Parameters:
#         image_path (str): 이미지 파일 경로
    
#     Returns:
#         str: 감지된 언어 코드 ('ko', 'en', 'ja', 'zh')
#     """
#     try:
#         # 이미지 열기
#         image = Image.open(image_path)
        
#         # 이미지가 RGBA인 경우 RGB로 변환
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')
            
#         # OCR 행 (언어 감 모드)
#         text = pytesseract.image_to_string(image, lang='kor+eng+jpn+chi_sim')
        
#         if not text.strip():
#             logging.warning(f"No text detected in image: {image_path}")
#             return 'unknown'
            
#         try:
#             # langdetect를 사용하여 언어 감지
#             detected_lang = detect(text)
            
#             # 언어 코드 매핑
#             lang_map = {
#                 'ko': 'ko',
#                 'ja': 'ja',
#                 'zh': 'zh',
#                 'en': 'en'
#             }
            
#             return lang_map.get(detected_lang, 'unknown')
            
#         except Exception as e:
#             logging.warning(f"Language detection failed for {image_path}: {str(e)}")
#             return 'unknown'
            
#     except Exception as e:
#         logging.error(f"Error processing image for language detection {image_path}: {str(e)}")
#         return 'unknown'
 


def config_reading(config_file: str) -> Dict:
    """설정 을 읽어서 반환하는 함수"""
    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"설정 파일을 불러오는데 실패했습니다: {str(e)}")
        sys.exit(1)

def collect_image_files(directory: str, json_data: Dict) -> List[str]:
    """
        이미지 파일 수집 함수

        Parameters:
            directory (str): 이미지 파일이 있는 디렉토리 경로
            json_data (Dict): 설정 파일 데이터
            
        Returns:
            List[str]: 이미지 파일 경로 리스트
    """
    image_files = []
    supported_formats = set(json_data['datafilter']['image_extensions'])
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                try:
                    # 파일명과 경로를 유니코드로 처리
                    file = os.path.normpath(file)
                    full_path = os.path.normpath(os.path.join(root, file))
                    
                    # 파일 확장자 체크
                    ext = os.path.splitext(file)[1].lower()
                    if ext in supported_formats:
                        image_files.append(full_path)
                except Exception as e:
                    logging.error(f"파 처리 중 오류 발생: {str(e)}")
    except Exception as e:
        logging.error(f"디렉토리 탐색 중 오류 발생: {str(e)}")
        
    return image_files

def process_tiff(file_path: str) -> str:
    """TIFF 파일을 처리하는 함수"""
    try:
        with Image.open(file_path) as img:
            return pytesseract.image_to_string(img, lang="kor+eng")
    except Exception as e:
        logging.error(f"TIFF 처리 중 오류 발생: {str(e)}")
        return ""

def preprocess_image_worker(image_data: bytes) -> bytes:
    """이미지 전처리를 위한 워커 함수"""
    try:
        # 입력 데이터 검증
        if not image_data:
            logging.warning("빈 이미지 데이터")
            return image_data

        # 이미지 포맷 확인
        is_png = image_data[:8].find(b'PNG') != -1
        is_jpeg = image_data[:3] == b'\xff\xd8\xff'
        
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(image_data, np.uint8)
        
        # IMREAD_UNCHANGED 플래그로 변경하여 알파 채널 포함하여 읽기
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            # logging.warning("이미지 디코딩 실패 - 원본 반환")
            return image_data
            
        # 알파 채널 처리
        if len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA to RGB 변환
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            rgb_img = img
        else:
            # 그레이스케일 이미지는 그대로 사용
            rgb_img = img

        # 이미지 크기 확인
        height, width = rgb_img.shape[:2]
        if height <= 0 or width <= 0:
            logging.warning(f"유효하지 않은 이미지 크기: {width}x{height}")
            return image_data
            
        # 그레이스케일 변환
        try:
            if len(rgb_img.shape) == 3:
                gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = rgb_img
        except Exception as e:
            logging.warning(f"그레이스케일 변환 실패: {str(e)}")
            return image_data
        
        # 이미지 크기 조정
        try:
            max_size = 4000
            if height > max_size or width > max_size:
                ratio = max_size / max(height, width)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                if new_width > 0 and new_height > 0:
                    gray = cv2.resize(gray, (new_width, new_height), 
                                   interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.warning(f"리사이징 실패: {str(e)}")
            return image_data

        # 이미지 인코딩
        try:
            if is_png:
                encode_param = [
                    int(cv2.IMWRITE_PNG_COMPRESSION), 9,
                    int(cv2.IMWRITE_PNG_STRATEGY), 
                    cv2.IMWRITE_PNG_STRATEGY_DEFAULT
                ]
                _, optimized = cv2.imencode('.png', gray, encode_param)
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, optimized = cv2.imencode('.jpg', gray, encode_param)
                
            if optimized is None:
                logging.warning("이미지 인코딩 실패 - 원본 반환")
                return image_data
                
            return optimized.tobytes()
            
        except Exception as e:
            logging.warning(f"이미지 인코딩 실패: {str(e)}")
            return image_data
        
    except Exception as e:
        logging.error(f"이미지 전처리 중 오류 발생: {str(e)}")
        return image_data

async def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        logging.error("OCR 서버 URL을 입력해주세요")
        sys.exit(1)
        
    server_url = sys.argv[1]
    json_data = config_reading('config.json')
    setup_logging(json_data)
    
    port = json_data['ocr_info']['ocr_server_port']
    server_url = f"http://{server_url}:{port}"

    client = OCRClient(
        server_url=server_url,
        config_path="config.json"
    )
    source_path = os.path.join(json_data['root_path'], json_data['datainfopath']['target_path'])
    
    logging.info("EASY OCR PROCESS START")
    start_time = time.time()
    results = await client.process_directory(source_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    logging.info(f"총 처리된 파일 수: {len(results)}")
    logging.info(f"처리 소요 시간: {hours}시간 {minutes}분 {seconds}초")

if __name__ == "__main__":
    asyncio.run(main())