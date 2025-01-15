import logging
import json
import requests
import json
import os
import uuid
import time

class OCRProcessor:
    def __init__(self):
        self.file_uuid_map = {}
        self.uuid_file_map = {}
        
    def create_uuid_mapping(self, file_path: str) -> str:
        """
        파일 경로에 대한 UUID 생성 및 매핑
        
        Parameters:
            file_path: str
            
        Returns:
            str: UUID 문자열
        """
        if file_path in self.file_uuid_map:
            return self.file_uuid_map[file_path]
            
        new_uuid = f"{uuid.uuid4()}_{int(time.time())}"
        self.file_uuid_map[file_path] = new_uuid
        self.uuid_file_map[new_uuid] = file_path
        return new_uuid

def setup_logger(config_path: str) -> logging.Logger:
    """
    로깅 설정 함수
    
    Parameters:
        config_path: str
        
    Returns:
        logging.Logger: 로거 객체
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    logging_level = config['ocr_info'].get('logging_level', 'INFO').upper()

    logger = logging.getLogger("OCRClient")
    logger.setLevel(getattr(logging, logging_level, logging.INFO))

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, logging_level, logging.INFO))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

def load_config(config_path: str) -> dict:
    """
    Config 파일 로드 함수
    
    Parameters:
        config_path: str
        
    Returns:
        dict: Config 데이터
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def send_image_to_ocr_server(logger: logging.Logger, server_url: str, image_paths: list) -> dict:
    """
    OCR 서버로 여러 이미지를 전송하고 ���답을 반환
    
    Parameters:
        logger: logging.Logger
        server_url: str
        image_paths: list
        
    Returns:
        dict: 이미지 전송 결과
    """
    if not image_paths:
        logger.error("전송할 이미지 파일이 없습니다.")
        return {"error": "No files to process"}

    # username = "ntsk"
    # password = "F@c4F@c5"
    # host = "172.10.13.123"
    # remote_path = "/data/upload/ocr_result"

    username = "nettars"
    password = "nettars1!"
    host = "118.37.61.92"
    port = "31221"
    remote_path = "/data/upload/ocr_result"
    
    results = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            logger.error(f"파일이 존재하지 않습니다: {image_path}")
            continue

        try:
            logger.debug(f"이미지 {image_path}를 서버 {server_url}로 전송합니다.")
            
            # JSON 페이로드 구성
            payload = {
                "inputUri": f"sftp://{username}:{password}@{host}:/{remote_path}/{os.path.basename(image_path)}",
                "outputUri": f"sftp://{username}:{password}@{host}:{port}/output/{os.path.splitext(os.path.basename(image_path))[0]}_output.json",
                "taskName": "textsense-test",
                "option": {
                    "withoutConversion": True
                },
                "extraJobs": [
                    {
                        "type": "textSense",
                        "option": {
                            "url": "http://hq.epapyrus.com:11056/textsense/api/job/build",
                            "reqType": "Document",
                            "reqOption": {
                                "SearchAPI": [
                                    {
                                        "Type": "FindRegion",
                                        "Name": "test1",
                                        "Key": [
                                            [103.79577464788731, 27.91146881287729],
                                            [179.24396378269623, 56.97786720321933]
                                        ]
                                    },
                                    {
                                        "Type": "FindRegion",
                                        "Name": "test2",
                                        "Key": [
                                            [181.42454728370228, 26.167002012072373],
                                            [274.31740442655985, 59.594567404426414]
                                        ]
                                    }
                                ]
                            },
                            "outputType": "json"
                        }
                    }
                ]
            }

            # JSON 데이터로 POST 요청
            response = requests.post(
                server_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.debug(f"서버로부터 {image_path}에 대한 응답을 성공적으로 받았습니다.")
                try:
                    response_data = response.json()
                    results.append({
                        'file_path': image_path,
                        'success': True,
                        'response': response_data
                    })
                except json.JSONDecodeError as je:
                    logger.error(f"JSON 파싱 오류 ({image_path}): {je}")
                    results.append({
                        'file_path': image_path,
                        'success': False,
                        'error': f"JSON 파싱 오류: {str(je)}"
                    })
            elif response.status_code == 400:
                logger.error(f"잘못된 요청 ({image_path}): {response.text}")
                results.append({
                    'file_path': image_path,
                    'success': False,
                    'error': f"��못된 요청: {response.text}"
                })
            elif response.status_code == 500:
                logger.error(f"서버 내부 오류 ({image_path}): {response.text}")
                results.append({
                    'file_path': image_path,
                    'success': False,
                    'error': f"서버 오류: {response.text}"
                })
            else:
                logger.error(f"예상치 못한 응답 코드 {response.status_code} ({image_path}): {response.text}")
                results.append({
                    'file_path': image_path,
                    'success': False,
                    'error': f"예상치 못한 응답 코드: {response.status_code}"
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"서버 전송 중 오류 발생 ({image_path}): {e}")
            results.append({
                'file_path': image_path,
                'success': False,
                'error': str(e)
            })

    return results

def main():
    """
    메인 함수
    """
    
    config_path = "config.json"
    
    # OCRProcessor 인스턴스 생성
    processor = OCRProcessor()
    
    # Config 로드
    json_data = load_config(config_path)
    logger = setup_logger(config_path)
    server_url = json_data['ocr_info']['server_url']

    # 소스 경로와 타겟 경로 가져오기
    source_path = json_data['ocr_info']['source_path']
    target_path = json_data['ocr_info']['target_path']
    image_extensions = json_data['ocr_info']['image_extension']

    # 경로 결합
    scan_path = os.path.join(source_path, target_path)
    
    # 이미지 파일 리스트 생성
    image_files = []
    for root, _, files in os.walk(scan_path):
        # 빈 폴더 건너뛰기
        if not files:
            continue
            
        for file in files:
            try:
                # 파일명과 경로에 특수문자가 있어도 안전하게 처리
                file_path = os.path.abspath(os.path.join(root, file))
                
                # 파일 크기가 0인 경우 건너뛰기
                if os.path.getsize(file_path) == 0:
                    logger.debug(f"빈 파일 건너뛰기: {file_path}")
                    continue
                    
                if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                    # 경로가 유효한지 확인
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        image_files.append(file_path)
                    else:
                        logger.warning(f"유효하지 않은 파일 경로: {file_path}")
            except Exception as e:
                logger.error(f"파일 경로 처리 중 오류 발생: {file} - {str(e)}")

    logger.info(f"전체 이미지 파일 수: {len(image_files)}")
    
    response = processor.send_image_to_ocr_server(logger, server_url, image_files)

    # 서버 응답 로그 출력
    logger.info("서버 응답:")
    logger.info(json.dumps(response, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()


