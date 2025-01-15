import os
import time
import logging
import inspect
from datetime import datetime
import builtins
import json
import inspect
import threading


from meta_info import CountersWithLock # 모듈에서 클래스를 가져옴

counter_log = CountersWithLock(1)


# 락 객체 생성
config_lock = threading.Lock()
 
logger = None   
# log_info.py
def config_reading():
    current_directory = os.getcwd()

    # JSON 파일 경로 생성
    print("[log_info] current_directory : ", current_directory)
    config_file = os.path.join(current_directory, 'config.json')

    if os.path.isfile(config_file):
        with config_lock:  # 락을 사용하여 동시에 여러 스레드가 접근하지 못하도록 보호
            with open(config_file, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                return json_data
    else:
        print("config.json 파일을 찾을 수 없습니다.")
        return None
    
set_json = config_reading()

print_message_toggle = set_json["print_message_toggle"]
debug_flag = set_json["debug_flag"]
log_to_file = set_json["log_to_file"]
log_to_console = set_json["log_to_console"]

log_to_level = set_json["log_to_level"]

# print("[log_info] print_message_toggle : ", print_message_toggle)
# print(" debug_flag : ", debug_flag)
# print(" log_to_file : ", log_to_file)
# print(" log_to_console : ", log_to_console)

# print(" log_to_level : ", log_to_level)

loginfo_path = set_json['log_path']
datainfopath = set_json['datainfopath']
source_path = datainfopath['source_path']
target_path = datainfopath['target_path']

loginfo_path = os.path.join(loginfo_path, source_path)

# 기본 'print' 함수 저장
original_print = builtins.print if 'print' in builtins.__dict__ else None

def toggle_printer(on=True):
    if on:
        # 'print' 함수가 있을 경우 기본 'print' 함수로 복원
        if original_print:
            builtins.print = original_print
    else:
        # 'print' 함수를 무시하는 람다 함수로 설정
        builtins.print = lambda *args, **kwargs: None
        
        
def debug_print(*args,  **kwargs):
    
    toggle_printer(on=debug_flag)
    
    # 스택 프레임 가져오기 (호출한 함수 정보 포함)
    caller_frame = inspect.currentframe().f_back
    
    # 호출한 함수 정보 출력
    caller_function = caller_frame.f_code.co_name
    caller_filename = caller_frame.f_code.co_filename
    caller_lineno = caller_frame.f_lineno
    
    # 현재 시간 기록
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    #print(f"Caller function: {caller_function}")
    # print(f"Caller filename: {caller_filename}")
    # print(f"Caller line number: {caller_lineno}")  
      
    for arg in args:
        print(arg)

    for key, value in kwargs.items():
        print(f"{key}: {value}")
    toggle_printer(False)
  
def process_status_info_print(*args,  **kwargs):
    
    toggle_printer(True)
    
    # 스택 프레임 가져오기 (호출한 함수 정보 포함)
    caller_frame = inspect.currentframe().f_back
    
    # 호출한 함수 정보 출력
    caller_function = caller_frame.f_code.co_name
    caller_filename = caller_frame.f_code.co_filename
    caller_lineno = caller_frame.f_lineno
    
    # 현재 시간 기록
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    #print(f"Caller function: {caller_function}")
    # print(f"Caller filename: {caller_filename}")
    #print(f"Caller line number: {caller_lineno}")  
    
    for arg in args:
        print(arg)

    for key, value in kwargs.items():
        print(f"{key}: {value}")
    toggle_printer(False)
                  
def status_info_print(*args,  **kwargs):
    
    toggle_printer(True)
    
    # 스택 프레임 가져오기 (호출한 함수 정보 포함)
    caller_frame = inspect.currentframe().f_back
    
    # 호출한 함수 정보 출력
    caller_function = caller_frame.f_code.co_name
    caller_filename = caller_frame.f_code.co_filename
    caller_lineno = caller_frame.f_lineno
    
    # 현재 시간 기록
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    #print(f"Caller function: {caller_function}")
    # print(f"Caller filename: {caller_filename}")
    #print(f"Caller line number: {caller_lineno}")  
      
    for arg in args:
        print(arg)
        # counter_log.add_log_file_list(arg) 

    for key, value in kwargs.items():
        print(f"{key}: {value}")
        # counter_log.add_log_file_list(f"{key}: {value}") 
    toggle_printer(False)
    
def setup_logger(name, log_file, log_level=logging.DEBUG, log_to_file=True, log_to_console=True):
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # level을 log_level로 수정

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 파일 핸들러 설정 (선택적)
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (출력을 터미널에도 함)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def setup_logger_info(name, log_file, log_level=logging.DEBUG, log_to_file=True, log_to_console=True):
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # level을 log_level로 수정

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 파일 핸들러 설정 (선택적)
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (출력을 터미널에도 함)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# 로그 설정
# log_to_file = False
# log_to_console = False
current_time = time.strftime("%Y_%m_%d%H%M%S", time.localtime())
process_log_msg_file_name = f"pre_process_log_msg_{current_time}.log"
process_log_info_file_name = f"pre_process_log_info_{current_time}.log"
ocr_process_log_file_name = f"ocr_process_log_{current_time}.log"

if log_to_level == 'logging.DEBUG':
    log_to_level =logging.DEBUG
elif log_to_level == "logging.INFO":
    log_to_level =logging.INFO
elif log_to_level == "logging.WARNING":
    log_to_level =logging.WARNING
elif log_to_level == "logging.ERROR":
    log_to_level =logging.ERROR
elif log_to_level == "logging.CRITICAL":
    log_to_level =logging.CRITICAL
        
if not os.path.exists(loginfo_path):
   os.makedirs(loginfo_path)

logger=setup_logger("pre_process", process_log_msg_file_name, log_level=log_to_level, log_to_file=log_to_file, log_to_console=log_to_console)

log_path = os.path.join(loginfo_path, process_log_info_file_name)

info_logger=setup_logger_info("pre_process_info", log_path , log_level=logging.DEBUG, log_to_file=log_to_file, log_to_console=log_to_console)

# 로그 함수 수정
def log_with_function_name(logger, message, log_level=logging.INFO):
    # 현재 함수의 이름 가져오기
    caller_function_name = inspect.currentframe().f_back.f_code.co_name
    # 로그 레벨에 따라 로그 함수 선택
    if log_level == logging.DEBUG:
        logger.debug(f"{caller_function_name}: {message}")
    elif log_level == logging.INFO:
        logger.info(f"{caller_function_name}: {message}")
    elif log_level == logging.WARNING:
        logger.warning(f"{caller_function_name}: {message}")
    elif log_level == logging.ERROR:
        logger.error(f"{caller_function_name}: {message}")
    elif log_level == logging.CRITICAL:
        logger.critical(f"{caller_function_name}: {message}")

# 예시 함수
# def test():
#     log_with_function_name(logger, "This is a test message", log_level=logging.INFO)

# if __name__ == "__main__":

#     toggle_printer(on=print_message_toggle)
#     setup_logger("pre_process", "pre_process_log_msg.log", log_level=logging.DEBUG, log_to_file=log_to_file, log_to_console=log_to_console)
    # test()
