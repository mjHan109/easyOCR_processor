import os
import time
import json
import logging  # logging 모듈을 import

import shutil
import re
import platform
import psutil
import sys
import socket
import threading
import zipfile
import tarfile
import rarfile
import datetime
import magic
import uuid

from openpyxl import load_workbook
from pptx import Presentation
from docx import Document
import log_info  # log_info 모듈을 임포트하여 설정을 공유

config_lock = threading.Lock()

D0_CF_11_list = ['.hwp', '.doc', '.xls', '.msg', '.ppt']
pk_list = ['.xlsx', '.pptx', '.docx', '.zip', '.jar']
bm_list = ['.bmp', '.dib']
ff_d8_ff_list = ['.jpg', '.jpeg', '.jfif', '.heic', '.jpe', '.hif']
# eml_header_list = ["b'Return-P'", "b'MIME-Ver'", "b'Delivere'", "b'From'", "b'ARC-Seal'", "b'Received'", "b'X-Sessio'", "b'MIME-Ver'", "b'Date: Mo'", "b'Date: Tu'", "b'Date: We'", "b'Date: Th'", "b'Date: Fr'", "b'X-Sessio'", "b'DKIM-Sig'"]
eml_header_list = [b'Return-P', b'MIME-Ver', b'Delivere', b'From', b'ARC-Seal', b'Received', b'X-Sessio', b'MIME-Ver', b'Date: Mo', b'Date: Tu', b'Date: We', b'Date: Th', b'Date: Fr', b'X-Sessio', b'DKIM-Sig']


class FileEncryptChecker:
    
    def __init__(self, file_path):
        self.file_path = file_path
        
    def is_file_encrypted(self):
        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()
        if ext == '.zip':
            return self.is_zip_encrypted()
        elif ext == '.rar':
            return self.is_rar_encrypted()
        elif ext == '.7z':
            return self.is_7z_encrypted()
        elif ext == '.xlsx':
            return self.is_excel_file_encrypted()
        elif ext == '.pptx':
            return self.is_ppt_encrypted()
        elif ext == '.docx':
            return self.is_docx_encrypted()
        else:
            print("Unsupported file format.")
            return False
    def is_zip_encrypted(self):
        try:
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                zip_ref.extractall(path='temp', pwd=None)
            return False
        except RuntimeError:
            return True
    def is_rar_encrypted(self):
        try:
            with rarfile.RarFile(self.file_path, 'r') as rar_ref:
                rar_ref.extractall(path='temp', pwd=None)
            return False
        except rarfile.BadRarFile:
            return True
    # def is_7z_encrypted(self):
    #     try:
    #         with py7zr.SevenZipFile(self.file_path, 'r', password=None) as z7_ref:
    #             z7_ref.extractall(path='temp')
    #         return False
    #     except py7zr.Bad7zFile:
    #         return True
    def is_excel_file_encrypted(self):
        try:
            with open(self.file_path, 'rb') as file:
                file_header = file.read(8)
                if file_header == b'\x50\x4B\x03\x04\x14\x00\x06\x00':
                    load_workbook(self.file_path)
                    return False
                else:
                    return True
            return False
        except Exception:
            return True
    def is_ppt_encrypted(self):
        try:
            with open(self.file_path, 'rb') as file:
                file_header = file.read(8)
                if file_header == b'\x50\x4B\x03\x04\x14\x00\x06\x00':
                    Presentation(self.file_path)
                    return False
                else:
                    return True
        except ValueError:
            return True
    def is_docx_encrypted(self):
        try:
            with open(self.file_path, 'rb') as file:
                file_header = file.read(8)
                if file_header == b'\x50\x4B\x03\x04\x14\x00\x06\x00':
                    Document(self.file_path)
                    return False
                else:
                    return True
        except Exception:
            return True
        
def check_signiture_check(file_ext, mime_type):
    if "txt" == file_ext:
        if "text" in mime_type:
            pass
        else:
            return True
    elif "xlsx" == file_ext:
        if "sheet" in mime_type:
            pass
        else:
            return True   
    elif "xls" == file_ext:
        if "excel" in mime_type:
            pass
        else:
            return True    
    elif "csv" == file_ext:
        if "csv" in mime_type:
            pass
        else:
            return True                                   
    elif "hwp" == file_ext:
        if "hwp" in mime_type:
            pass
        else:
            return True   
    elif "pdf" == file_ext:
        if "pdf" in mime_type:
            pass
        else:
            return True   
    elif "pptx" == file_ext:
        if "presentation" in mime_type:
            pass
        else:
            return True          
    elif "docx" == file_ext:
        if "document" in mime_type:
            pass
        else:
            return True                        
    return False

#//////////////////////////////////////////////////////////////////////////////////////////          
def display_filter_config_info(config_info_json):
    # config_info_json =  main_ut.config_reading('config.json')

    # data filter info
    datafilter = config_info_json['datafilter']

    filter_info_list = []

    allfilterinfolist = []
    filter_extensions= datafilter['filter_extensions']
    analyzerinfolist = datafilter['analyzer_extensions']
    imageinfolist = datafilter['image_extensions']
    email_extensions = datafilter['email_extensions']
    compression_extensions = datafilter['compression_extensions']
    file_read_extensions = datafilter['file_read_extensions']


    tika_app = config_info_json['tika_app']
    tika_server_mode = tika_app['tika_server_mode']
    tika_server_ip = tika_app['tika_server_ip']
    tika_server_port = tika_app['tika_server_port']
            
    log_info.status_info_print( f"analyzerinfolist: {analyzerinfolist}")
    log_info.status_info_print( f"imageinfolist: {imageinfolist}" )
    log_info.status_info_print( f"email_extensions: {email_extensions}" )
    log_info.status_info_print( f"compression_extensions: {compression_extensions}" )
    log_info.status_info_print( f"file_read_extensions: {file_read_extensions}" )

    allfilterinfolist.extend(filter_extensions)
    allfilterinfolist.extend(analyzerinfolist)
    allfilterinfolist.extend(imageinfolist)
    allfilterinfolist.extend(email_extensions)
    allfilterinfolist.extend(compression_extensions)   
    
    filter_info_list.append(allfilterinfolist)
    filter_info_list.append(compression_extensions)
    filter_info_list.append(analyzerinfolist)
    filter_info_list.append(file_read_extensions)
    filter_info_list.append(email_extensions)                
    filter_info_list.append(imageinfolist)  
    
    log_info.status_info_print(f" filter_extensions : {datafilter['filter_extensions']}")
    
    return  filter_info_list

def check_and_print_params():
    """
    sys.argv로 받은 파라미터를 체크하고 결과를 출력하는 함수
    """
    revalue1 =  False
    revalue2 =  False
    revalue3 =  False
    revalue4 =  False
    revalue5 =  False

    if len(sys.argv) == 6:
        arg_value1 = sys.argv[1]
        arg_value2 = sys.argv[2]
        arg_value3 = sys.argv[3]
        arg_value4 = sys.argv[4]
        arg_value5 = sys.argv[5]
    else:
        log_info.status_info_print(f"오류: 파라미터는 5개여야 합니다. len is {len(sys.argv)}")
        sys.exit(1)

    check_and_print_result("pst to eml", arg_value1)
    check_and_print_result("EDB", arg_value2)
    check_and_print_result("NSF", arg_value3)
    check_and_print_result("OCR", arg_value4)
    check_and_print_result("add", arg_value5)
    
    if arg_value1 == "Y" or arg_value1 == "y":
        revalue1 =  True
    if arg_value2 == "Y" or arg_value1 == "y":
        revalue2 =  True
    if arg_value3 == "Y" or arg_value1 == "y":
        revalue3 =  True
    if arg_value4 == "Y" or arg_value1 == "y":
        revalue4 =  True
    if arg_value5 == "Y" or arg_value1 == "y":
        revalue5 =  True        
               
    return revalue1, revalue2, revalue3, revalue4, revalue5

def check_and_print_result(label, value):
    """
    파라미터를 체크하고 결과를 출력하는 함수
    """
    if value == 'Y':
        log_info.status_info_print( f"{label}, checking")
    else:
        log_info.status_info_print( f"{label}, no checking")
# 바이트를 기가바이트로 변환하는 함수
def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)
       
# 현재 CPU 사용률 가져오기
def config_reading(json_file_name):
    current_directory = os.getcwd()

    # JSON 파일 경로 생성
    config_file = os.path.join(current_directory, json_file_name)
    
    # log_info.status_info_print( f" config_reading : {current_directory} - {config_file} ")     
    if os.path.isfile(config_file):
        with config_lock:  # 락을 사용하여 동시에 여러 스레드가 접근하지 못하도록 보호
            with open(config_file, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                return json_data
    else:
        log_info.status_info_print( f"{current_directory} - {json_file_name} 파일을 찾을 수 없습니다.")
        return None
    
filter_header_list = config_reading("filter_header.json")
# log_info.status_info_print(f"filter_header_list : {filter_header_list}")



def read_file_from_path(file_path, counters, mode):
    log_info.debug_print( f" read_file_from_path : {file_path} ")
    try:
        # 파일의 상세한 메타데이터 가져오기
        file_stats = os.stat(file_path)

        # 파일의 크기 가져오기
        file_size = file_stats.st_size

        # 파일의 생성 시간 가져오기
        creation_time = time.ctime(file_stats.st_ctime)

        # 파일의 최근 수정 시간 가져오기
        modification_time = time.ctime(file_stats.st_mtime)

        # 파일의 최근 접근 시간 가져오기
        access_time = time.ctime(file_stats.st_atime)

        # 파일의 inode 번호 가져오기
        inode_number = file_stats.st_ino

        # 파일의 권한 모드 가져오기
        mode = file_stats.st_mode

        # 파일의 장치 번호 가져오기
        device = file_stats.st_dev

        # 파일의 UID 가져오기
        uid = file_stats.st_uid

        # 파일의 GID 가져오기
        gid = file_stats.st_gid

        # 파일의 소유자 정보 가져오기
        owner_info = f"{uid}:{gid}"
        
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)  
          
        body_info = None
        if mode:
        # 파일 내용 가져오기
            with open(f"{file_path}", 'r', encoding='utf-8', errors='ignore') as file:
                body_info = file.read()
        # text 출력
        log_info.debug_print(body_info)        
        # JSON 객체 생성
        meta_info = {
            "accessed": access_time,
            "created":creation_time,
            # "ctime": None,
            "mtime": modification_time,
            "size": file_size,
            "owner": owner_info,
            "mime_type":mime_type
        }

        # JSON 객체 출력
        log_info.debug_print(json.dumps(meta_info, indent=4))    
        file_size  = file_size / 1024
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        info_message = f"file read info: {file_path} : filesize : {file_size} kb- {current_time}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.DEBUG)
        # log_info.status_info_print(body_info)
        return meta_info, body_info
    except Exception as e:      
        info_message = f"{file_path}, a exceptions: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        counters.increment_files_exception_count()
        return None, None
  

def es_indexing_ext(es, jsondata, main_fold, file_path, exc_file_type, es_mng, counters):
    
    now = datetime.datetime.now()
    if file_path is None:
        return
    
    docmetainfo = {}
    metadata_info = None
    text = None 
    step_num = 99
    docmetainfo["tags"] = []    
    file_path = f"""{file_path}"""
    
    try:                 
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        file_ext = file_ext.lower() 
        new_uuid = uuid.uuid4()
        utc_now = datetime.datetime.utcnow()
        timestamp = utc_now.timestamp()
        uuid_str = str(new_uuid) + '_' + str(timestamp)
            
        if os.name == 'nt':
            splitdrive_file_path = os.path.dirname(file_path)
        else:
            splitdrive_file_path = os.path.dirname(file_path)

        file_size = os.path.getsize(file_path) 

        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)
        step_num = 77
        
    except Exception as e:
        info_message = f" : {file_path}, Exception 0: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
                
    docmetainfo["file"] = {}
    try:

        docmetainfo["root_path"] = main_fold 
        docmetainfo["directory"] = splitdrive_file_path
        docmetainfo["title"] = file_name
        docmetainfo["file"]["path"] = f"""{file_path}"""
        docmetainfo["file"]["extension"] = file_ext[1:]
        docmetainfo["file"]["size"] = file_size
        docmetainfo["uuid"] = uuid_str 
        # log_info.process_status_info_print(f"aaaa start : {file_type} - {main_fold} - {sindex} - {allcount} : {file_path}")
        docmetainfo["tags"].append("file")
        docmetainfo["tags"].append("exception")


        docmetainfo["@timestamp"] = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        # 파일 처리
        docmetainfo["file"]["type"] = "file"
        docmetainfo["file"]["mime_type"] = f"{exc_file_type}"
    except Exception as e:
        info_message = f" : {file_path}, Exception 1: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
        return 

    # 파일 처리

    # try:

    #     metadata_info, text = read_file_from_path(file_path, counters, False)

    # except Exception as e:
    #     info_message = f" : {file_path}, Exception read_file_from_path: {str(e)}"
    #     log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
    
    # try:
            
    #     if metadata_info is None:
    #         return 
    #     else:
    #         step_num = 1    
    #         try:
    #             step_num = 2
    #             docmetainfo["file"]["accessed"] = metadata_info["accessed"]
    #         except KeyError:
    #             step_num = 3
    #             pass
    #         step_num = 4  
    #         try:
    #             # docmetainfo["file"]["created"] = metadata_info["created"]  
    #             docmetainfo["date"] = metadata_info["created"]  
    #         except KeyError:
    #             pass
    #         # try:
    #         #     docmetainfo["file"]["ctime"] = metadata_info["ctime"]
    #         # except KeyError:
    #         #     pass
    #         step_num = 5  
    #         try:
    #             docmetainfo["file"]["mtime"] = metadata_info["mtime"]
    #         except KeyError:
    #             pass
    #         step_num = 6   
    #         try:
    #             docmetainfo["file"]["owner"] = metadata_info["owner"]
    #         except KeyError:
    #             pass
    #         step_num = 7  
    #         try:
    #             docmetainfo["file"]["size"] = metadata_info["size"]
    #         except KeyError:
    #             pass 
    #         step_num = 8       
    #         docmetainfo["file"]["meta_info"] = f"""{metadata_info}""" 
    #         step_num = 9
    # except Exception as e:
    #     info_message = f" : {file_path}, Exception 1: {str(e)}"
    #     log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
    #     return  
                                            
    if es is not None: # indexing document
        # while True:
        step_num = 25
        es_status = es_mng.es_index_data(es, jsondata, 'el_index_main_name', docmetainfo, file_path, counters)
        step_num = 29
        
        if es_status == False:
            step_num = 27
            es = es_mng.es_index_connection(jsondata)
            
            if es is None:
                pass
                # sys.exit(1)       
            else:
                es_status = es_mng.es_index_data(es, jsondata, 'el_index_main_name', docmetainfo, file_path, counters)
                if es_status == False:
                    pass


def check_file_drm(file_path):
    with open(file_path, "rb") as file:
        file_data = file.read()
    # log_info.status_info_print(f"file_path : {file_data}\n\n\n")

    # 국가정보연구원 자료 참조 : DRMSignature.ini
    # Fasoo DRM 체크
    if (
        b"\x3C\x21\x2D\x2D\x20\x49\x4E\x43\x4F\x50\x53\x20\x53\x45\x43\x55\x2D\x44\x52\x4D" in file_data
        or b"\x3C\x21\x2D\x2D\x20\x46\x61\x73\x6F\x6F\x53\x65\x63\x75\x72\x65\x43\x6F\x6E\x74\x61\x69\x6E\x65\x72" in file_data
        or b"\x50\x73\x46\x68\x01\x00\xD0\x00\x00\x00" in file_data
        or b"\x9b\x20\x44\x52\x4d\x4f\x4e\x45" in file_data
    ):
        return "Fasoo DRM"
    # Markany DRM 체크
    if (
        b"\x8c\x17\x2e\x9a\x10\xb6\xf0\xd4\x9c\xa9\xca\xee\x80\xbc\x93\x9e" in file_data
        or b"\xC5\x63\x4A\x59\x8C\x1B\xCE\xBD\xBB\xDE\xC3\x59\x3E\xD1\x78\x1C" in file_data
        or b"\x3C\x4D\x41\x52\x4B\x41\x4E\x59\x5F\x44\x4F\x43\x55\x4D\x45\x4E\x54\x53\x41\x46\x45\x52\x3E" in file_data
        or b"\x3C\x44\x4F\x43\x55\x4D\x45\x4E\x54\x20\x53\x41\x46\x45\x52\x20\x56\x32\x30\x31\x37\x3E" in file_data
        or b"\x3C\x44\x4F\x43\x55\x4D\x45\x4E\x54\x53\x41\x46\x45\x52" in file_data
    ):
        return "Markany DRM"
    # Softcamp DRM 체크
    if (
        b"\x53\x43\x44\x53\x41\x30\x30" in file_data
        or b"\x53\x43\x44\x53\x53\x53\x30" in file_data
    ):
        return "Softcamp DRM"
    return None

    
def drm_check_file(file_path):
    drm_type = check_file_drm(file_path)
    if drm_type is None:
        return None
    else:
        filter_ext = identify_file_type_for_error(file_path)
        info = {
                "file_path": f"""{file_path}""",
                'filter_ext':f"""{filter_ext}""",
                "error_message": f"""drm:{drm_type}"""
        }
        return info
            
def get_cpu_usage():
    return psutil.cpu_percent(interval=0.001)

# 메모리 사용량 체크 함수
def check_memory_usage(threshold_percent=70):
    
    memory_percent = psutil.virtual_memory().percent
    info_message = (f"memory_percent : {memory_percent}")
    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)
    return memory_percent <= threshold_percent

def get_max_kernel_thread_num():
    try:
        with open("/proc/sys/kernel/threads-max", "r") as file:
            threads_max_value = int(file.read().strip())
            log_info.status_info_print(f"/proc/sys/kernel/threads-max 값: {threads_max_value}")
        return threads_max_value
    except FileNotFoundError:
        log_info.status_info_print("/proc/sys/kernel/threads-max 파일을 찾을 수 없습니다.")
    except Exception as e:
        log_info.status_info_print("오류 발생:", str(e))
    return 0

def identify_file_type_new(file_path, file_ext, counters):
    try:    
        with open(file_path, 'rb') as file:
            header = file.read(8)  # 파일의 처음 8바이트를 읽음
            # log_info.status_info_print( f"{header}: {file_path}: {header}")
            
        ext = file_ext
        step = 0
        
        try:
            filter_header = filter_header_list.get(file_ext)
            # log_info.status_info_print(f"filter_header : {filter_header}")
            if filter_header is None:
                info_message = f" there is no header info {header} : {file_path} - {file_ext}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.WARNING)
                return False
            else:                
                if  header in filter_header: 
                    ext = None
                    info_message = f" there is filtered info {header} : {filter_header} - {file_path} - {file_ext}"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
                    return False                  
                else:
                    # info_message = f" there is no filtered info {header} : {filter_header} - {file_path} - {file_ext}"
                    # log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
                    return True

                
        except Exception as e:
            step = 11
            # counters.increment_files_exception_count()
            info_message = f"Exception header Info {step} -  {header}: {file_ext} - {ext} {str(e)}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
            return False
        
    except Exception as e:
        step = 100
        # counters.increment_files_exception_count()
        info_message = f"Exception 0 {step} -  {file_ext} - {ext} {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
        return False
          
def identify_file_type(file_path, file_ext, counters):
    try:    
        with open(file_path, 'rb') as file:
            header = file.read(8)  # 파일의 처음 8바이트를 읽음
            # log_info.status_info_print( f"{header}: {file_path}: {header}")
            
        ext = ""
        step = 0
        
        try:
            if file_ext == '.eml':
                step = 1
                if  header in eml_header_list: # ["b'From'", "b'ARC-Seal'", "b'Received'", "b'X-Sessio'", "b'MIME-Ver'", "b'Date: Mo'", "b'Date: Tu'", "b'Date: We'", "b'Date: Th'", "b'Date: Fr'", "b'X-Sessio'", "b'DKIM-Sig'"]
                    ext = '.eml'
                else:
                    ext = None
            # ['.hwp', 'doc', 'xls', '.msg'] 
            elif header.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1') or  header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                step = 2
                if file_ext in D0_CF_11_list: 
                    ext = file_ext
                else:
                    ext = None
            # ['.xlsx', '.pptx', 'docx', '.zip', 'jar']
            elif header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x03\x04\x14\x00\x06\x00') or header.startswith(b'\x50\x4B\x03\x04'):
                step = 3
                if file_ext in pk_list: 
                    ext = file_ext
                else:
                    ext = None
            # ['.bmp', 'dib']
            elif header.startswith(b'BM') or header.startswith(b'\x42\x4D'):
                step = 4
                if file_ext in bm_list: 
                    ext = file_ext
                else:
                    ext = None
            # ['.jpg', '.jpeg', '.jfif', 'heic', '.jpe', 'hif'] 
            elif header.startswith(b'\xff\xd8\xff\xe0') or header.startswith(b'\xff\xd8\xff\xe1') or header.startswith(b'\xff\xd8\xff\xe10') or header.startswith(b'IF') or header.startswith(b'\x49\x46') or header.startswith(b'if') or header.startswith(b'\x69\x66'):
                step = 5
                if file_ext in ff_d8_ff_list: 
                    ext = file_ext
                else:
                    ext = None
            elif header.startswith(b'HWP Docu'): 
                step = 6
                if file_ext == '.hwp':
                    ext = '.hwp'
                else:
                    ext = None
                    
            elif header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1') or header.startswith(b'\xfd\xff\xff\xff\x10\x1f"##()') or header.startswith(b'\xfd\xff\xff\xff\x10\x1f"\x02') or header.startswith(b'\xfd\xff\xff\xff\x00\x00\x00') or header.startswith(b'\x09\x05\x06\x00') or header.startswith(b'<HTML>\r\n') or header.startswith(b'      <m'):
                step = 7
                if file_ext == '.xls':
                    ext = '.xls'
                else:
                    ext = None
                    
            elif  header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1') or header.startswith(b'\x00n\x1e\xf0') or header.startswith(b'\x0f\x00\xe8\x03') or header.startswith(b'\xa0F\x1d\xf0') or header.startswith(b'\xfd\xff\xff\xff\x0e\x1cC\x00\x00\x00') or header.startswith(b'\x00\x6E\x1E\xF0'):
                step = 8
                if file_ext == '.ppt':
                    ext = '.ppt'
                else:
                    ext = None
            elif header.startswith(b'\x0dDOC') or header.startswith(b'1\xbe\x00\x00\x00\xab') or header.startswith(b'\x9b\xa5') or header.startswith(b'\xcf\x11\xe0\xa1\xb1\x1a\xe1\x00') or header.startswith(b'\xdb\xa5') or header.startswith(b'\xec\xa5\xc1\x00'):
                step = 9
                if file_ext == '.doc':
                    ext = '.doc'
                else:
                    ext = None
            elif header.startswith(b'\x25\x50\x44\x46') or header.startswith(b'%PDF-') or  header.startswith(b'\x00\x05\x16\x07\x00\x02\x00\x00'):
                step = 10
                if file_ext == '.pdf':
                    ext = '.pdf'
                else:
                    ext = None
            elif header.startswith(b'Inno Set') or header.startswith(b'l\x02\x00\x00\x0c,\x00\x00') or header.startswith(b'h\x02\x00\x00T+\x00\x00'):
                step = 11
                if file_ext == '.msg':
                    ext = '.msg'
                else:
                    ext = None
            elif header.startswith(b'PKLITE') or header.startswith(b'PKSpX') or header.startswith(b'WinZip'):
                step = 12
                if file_ext == '.zip':
                    ext = '.zip'
                else:
                    ext = None
            elif header.startswith(b'Rar!\x1a\x07\x00'):
                step = 13
                ext = '.rar'
            elif header.startswith(b'\x1f\x9d\x90'):
                step = 14
                ext = '.tar.z'
            elif header.startswith(b'BZh'):
                step = 15
                ext = 'tar'
            elif header.startswith(b'\x1f\x8b\x08'):
                step = 16
                ext = '.gz'
            elif header.startswith(b"7z\xbc\xaf'\x1c"):
                step = 17
                ext = '.7z'
            elif header.startswith(b'-lh') or header.startswith(b'-lh5-'):
                step = 18
                if file_ext == '.lzh':
                    ext = '.lzh'
                else:
                    ext = None
            elif header.startswith(b'\x21\x42\x44\x4E') or header.startswith (b'!BDN') or header.startswith(b'\xec\x06\x00\x00\x01\x00\x00\x00'):
                step = 19
                if file_ext == '.pst':
                    ext = '.pst'
                else:
                    ext = None
            elif header.startswith(b'IFF\x00') or header.startswith(b'\xff\xd8\xff'):
                step = 21
                if file_ext == '.jpg':
                    ext = '.jpg'
                else:
                    ext = None
            elif header.startswith(b'GIF87a') or header.startswith(b'\x47\x49\x46\x38\x37\x61') or header.startswith(b'GIF89a') or header.startswith(b'\x47\x49\x46\x38\x39\x61'):
                step = 22
                if file_ext == '.gif':
                    ext = '.gif'
                else:
                    ext = None
            elif header.startswith(b'\x49\x49\x2A') or header.startswith(b'\x4D\x4D\x2A'):
                step = 23
                if file_ext == '.tif':
                    ext = '.tif'
                else:
                    ext = None 
            elif header.startswith(b'\x4D\x4D\x00\x2A') or header.startswith(b'\x4D\x4D\x00\x2B'):
                step = 24
                if file_ext == '.tiff':
                    ext = '.tiff'
                else:
                    ext = None 
            elif header.startswith(b'\x52\x49\x46\x46') or header.startswith(b'\x41\x56\x49\x20\x4C\x49\x53\x54') or header.startswith(b'AVI LIST'):
                step = 25 
                if file_ext == '.avi':
                    ext = '.avi'
                else:
                    ext = None 
            elif header.startswith(b'FORM\x00') or header.startswith (b'\x46\x4F\x52\x4D\x00'):
                step = 26
                if file_ext == '.aiff':
                    ext = '.aiff'
                else:
                    ext = None
            elif header.startswith(b'JARCS\x00') or header.startswith(b'\x4A\x41\x52\x43\x53\x00') or header.startswith(b'\x50\x4B\x03\x04'):
                step = 27
                if file_ext == '.jar':
                    ext = '.jar'
                else:
                    ext = None
            elif header.startswith(b'ID3') or header.startswith(b'\x49\x44\x33'):
                step = 28
                if file_ext == '.mp3':
                    ext = '.mp3'
                else:
                    ext = None
            elif header.startswith(b'moov') or header.startswith(b'\x6D\x6F\x6F\x76') or header.startswith(b'free') or header.startswith(b'\x66\x72\x65\x65') or header.startswith(b'wide') or header.startswith(b'\x77\x69\x64\0x65'):
                step = 29
                if file_ext == '.mov':
                    ext = '.mov'
                else:
                    ext = None
            elif header.startswith(b'{\\rtf1') or header.startswith(b'\x7B\x5C\x72\x74\x66\x31'):
                step = 30
                if file_ext == '.rtf':
                    ext = '.rtf'
                else:
                    ext = None
            elif header.startswith(b'RIFF') or header.startswith(b'\x52\x49\x46\x46') or header.startswith(b'WAVEfmt ') or header.startswith(b'\x57\x41\x56\x45\x66\x6D\x74\x20'):
                step = 31
                if file_ext == '.wav': 
                    ext = '.wav'
                else:
                    ext = None
            elif header.startswith(b'\x2A\x2A\x2A\x20\x20\x49\x6E\x73') or header.startswith(b'***  Ins') or header.startswith(b'tallatio') or header.startswith(b'\x74\x61\x6C\x6C\x61\x74\x69\x6F') or header.startswith(b'n Starte') or header.startswith(b'\x6E\x20\x53\x74\x61\x72\x74\x65') or header.startswith(b'd ') or header.startswith(b'\x64\x20'):
                step = 32
                if file_ext == '.log':
                    ext = '.log'
                else:
                    ext = None
                    
            elif header.startswith(b'Level,Ti') or header.startswith(b'\xb5\xb5\xb8\xc5') or header.startswith(b'\xb1\xd9\xb9') or header.startswith(b'\xbb\xe7\xbe\xf7\xc0\xda\xb9\xf8') or header.startswith(b'\xef\xbb\xbf\xeb\xb2\x88\xed\x98') or header.startswith(b'\xb0\xc5\xb7\xa1\xc0\xcf\xc0\xda') or header.startswith(b'\xef\xbb\xbf\xec\x97\x90\xeb\x9f') or header.startswith(b'2021\xb3\xe2 7') or header.startswith(b'\r\n\r\n\r\n\r\n'):
                step = 33
                if file_ext == '.csv':
                    ext = '.csv'
                else:
                    ext = None                     
            #  need update     
            elif header.startswith(b'<?xml version=') or header.startswith(b'<?xml ve') or header.startswith(b"<?xml "):
                step = 34
                if file_ext == '.svg':
                    ext = '.svg'
                elif file_ext == ".xml":
                    ext = '.xml'
                elif file_ext == ".html": 
                    ext = '.html'
                else: 
                    ext = None
            elif  header.startswith(b'From 169') or header.startswith(b'From') or header.startswith(b'\x46\x72\x6F\x6D\x20\x20\x20') or header.startswith(b'From ????') or header.startswith(b'\x46\x72\x6F\x6D\x20\x3F\x3F\x3F') or header.startswith(b'From: '):
                step = 35
                # log_info.status_info_print( f"{file_path}: {header}")
                if file_ext == '.mbox':
                    ext = '.mbox'
                else: 
                    None 
            elif file_ext == '.png': #header.startswith(b'\x89PNG\r\n\x1a\n') or header.startswith(b'PNG') or header.startswith(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'):        
            # elif header.startswith(b'\x89PNG\r\n\x1a\n') or header.startswith(b'PNG') or header.startswith(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'):
                step = 20
                if file_ext == '.png':
                    ext = '.png'
                else:
                    ext = None           
            elif header.startswith(b'0'):
                step = 36
                ext = '.ost'
            elif header.startswith(b'txt') or header.startswith(b'\xed\x95\x9c\xea\xb5\xad\xeb\x82'):
                step = 37
                ext = '.txt'
            # elif header.startswith(b'From') or header.startswith(b'\x46\x72\x6F\x6D\x20\x20\x20') or header.startswith(b'From ????') or header.startswith(b'\x46\x72\x6F\x6D\x20\x3F\x3F\x3F') or header.startswith(b'From: ') or header.startswith(b'\x46\x72\x6F\x6D\x3A\x20') or header.startswith(b'ARC-Seal') or header.startswith(b'Received'):
            #     if file_ext == '.eml':
            #         ext = '.eml'
            #     else:
            #         ext = None                                  
            else:
                step = 38
                info_message = f"header Info {header}: {file_path} - {file_ext}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
                ext = None
                
        except Exception as e:
            step = 11
            # counters.increment_files_exception_count()
            info_message = f"Exception header Info {step} -  {header}: {file_ext} - {ext} {str(e)}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
            
        if file_ext == ext:
            info_message = f"True header Info {step} - {header}: {file_path}{ext}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)
            return True
        else:
            info_message = f"False header Info step - {step} - {header}: {file_path} -  {file_ext}:{ext}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.WARNING)
            return False
    except Exception as e:
        step = 100
        # counters.increment_files_exception_count()
        info_message = f"Exception 0 {step} -  {file_ext} - {ext} {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)   
        
                
def identify_file_type_for_error(file_path):
    
    header = ""
    try:    
        with open(file_path, 'rb') as file:
            header = file.read(8)  # 파일의 처음 8바이트를 읽음     
            return header
        
    except Exception as e:
        step = 100
        # counters.increment_files_exception_count()
        info_message = f"Exception  {step} - {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
        return info_message  
    
def make_error_data(file_path, error_msg):
    # shutil.move(file_path, issue_path)
    filter_ext = identify_file_type_for_error(file_path)
    info = {}
    try:   
        if filter_ext is None:
            filter_ext = "no_ext"
        if file_path is None:
            file_path = "no_path_name"        
        info = {
                "file_path": f"""{file_path}""",
                'filter_ext':f"""{filter_ext}""",
                "error_message": f"""{error_msg}"""
        }
    except Exception as e:
        info_message = f"make_error_data -  {str(e)} "
        log_info.status_info_print(info_message)         
    # info_message = f"identify_file - {info}"
    # log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
    return info
                              
def check_duplicated_path(file_path):
    try:     
        base_name, extension = os.path.splitext(file_path)
        
        if os.path.exists(file_path):
            count = 1
            new_file_path = f"{base_name}_{count}{extension}"
            
            while os.path.exists(new_file_path):
                new_file_path = f"{base_name}_{count}{extension}"
                count += 1
                
            return new_file_path
        else:
            return file_path
    except Exception as e:  
        info_message = f"Exception: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return file_path  

def rename_folders(directory_path):
    # 지정한 디렉토리 내의 모든 폴더 리스트 가져오기
    folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]

    # 각 폴더의 이름 변경
    for folder in folders:
        
        new_name = replace_brackets_onely_file_name(folder)
        
        old_path = os.path.join(directory_path, new_name)
        new_path = os.path.join(directory_path, new_name)
        if old_path != new_path:
            # 이름 변경
            os.rename(old_path, new_path)
            log_info.status_info_print(f'폴더 "{old_path}"의 이름이 "{new_path}"으로 변경되었습니다.')


def make_fold_with_file_name_without_ext(file_path, output_dir, masking_path):
    # 파일의 기본 이름과 확장자 추출
    dir_path = os.path.splitext(file_path)[0]
    relative_path = os.path.relpath(dir_path, masking_path)    
    new_path_with_file_name = os.path.join(output_dir, relative_path)
    os.makedirs(new_path_with_file_name, exist_ok=True)
    return new_path_with_file_name

def get_file_name_without_ext(file_path):
    # 파일의 기본 이름과 확장자 추출
    file_name_with_ext = os.path.basename(file_path)
    file_name, exit = os.path.splitext(file_name_with_ext)
    # 제일 마지막 점 뒤의 문자열을 확장자로 반환 (소문자로 변환)
    file_name = file_name.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_').replace('~', '_').replace(',', '_').replace('.', '_').replace('「', '-').replace('」', '-')
    return file_name

def get_file_extension(file_path):
    # 파일의 기본 이름과 확장자 추출
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    # 제일 마지막 점 뒤의 문자열을 확장자로 반환 (소문자로 변환)
    return ext[1:].lower() 
           
def get_main_fold(masking_path, file_path):
    # root를 제외한 상대 경로 얻기
    relative_path = os.path.relpath(file_path, masking_path)

    # 상대 경로를 구성 요소로 분리
    components = os.path.normpath(relative_path).split(os.path.sep)

    # 최상위 폴더 얻기
    main_fold = components[0] if components else None
    return main_fold      

def get_mail_main_fold(masking_path, file_path):
    # root를 제외한 상대 경로 얻기
    relative_path = os.path.relpath(file_path, masking_path)

    # 상대 경로를 구성 요소로 분리
    components = os.path.normpath(relative_path).split(os.path.sep)

    # 최상위 폴더 이후의 다음 폴더 이름 얻기
    next_folder = components[3] if len(components) > 1 else None
    # log_info.status_info_print(f"main fold mail:{components}- {masking_path} : {next_folder} :  {relative_path} :  {file_path}")
    return next_folder

def replace_brackets_onely_file_name(file_name):
    try:      
        # 파일 이름에서 [ 또는 ]를 _로 대체
        file_name = file_name.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_').replace('~', '_').replace(',', '_').replace('.', '_').replace('「', '-').replace('」', '-')
        file_name = re.sub(r'[^a-zA-Z0-9가-힣_]', '', file_name)
        new_file_name = file_name
        return new_file_name
    
    except Exception as e:  
        info_message = f"Exception: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return file_name  
    
def replace_brackets_file_name_with_ext(file_name):
    try:      
        base_name, extension = os.path.splitext(file_name)
        
        # 파일 이름에서 [ 또는 ]를 _로 대체
        base_name = base_name.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_').replace('~', '_').replace(',', '_').replace('.', '_').replace('「', '-').replace('」', '-')
        base_name = re.sub(r'[^a-zA-Z0-9가-힣_]', '', base_name)
        new_file_name = base_name + extension
        
        return new_file_name
    except Exception as e:  
        info_message = f"Exception: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return file_name  

def replace_brackets_with_underscore_only_directroy_path_name(directory_path):
    try:  
        # 파일 이름에서 [ 또는 ]를 _로 대체
        directory_path = directory_path.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_').replace('~', '_').replace(',', '_').replace('.', '_').replace('「', '-').replace('」', '-')
        new_fdirectory_path = ''.join(e for e in directory_path if e.isalnum() or e in ('_', '/'))
        
        return new_fdirectory_path
    except Exception as e:  
        info_message = f"Exception: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return directory_path       
                    
def replace_brackets_with_underscore_file_name(file_path):
    try:      
        # 파일 경로에서 파일 이름만 추출
        file_name = os.path.basename(file_path)

        new_base_name = replace_brackets_file_name_with_ext(file_name)
        #directory path name
        path_name = replace_brackets_with_underscore_only_directroy_path_name(os.path.dirname(file_path))
        # 새로운 파일 이름을 포함한 전체 경로 생성
        # log_info.status_info_print( f"\n replace_brackets_with_underscore_file_name 1 -{file_path} -  {path_name} - {new_base_name}")
        new_file_path = os.path.join(path_name, new_base_name)
        # log_info.status_info_print( f"\n replace_brackets_with_underscore_file_name 2 - {new_file_path}")
        return new_file_path

    except Exception as e:  
        info_message = f"Exception: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return file_path
    
def replace_brackets_with_underscore_all_path(file_path):
    try:  
        new_file_path = file_path.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_').replace('~', '_')
        new_file_path = ''.join(e for e in new_file_path if e.isalnum() or e in ('_', '.', '/'))
        
        return new_file_path

    except Exception as e:  
        info_message = f"Exception: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        return file_path
    
def rename_file_with_underscore(original_path, counters):
    # [ 또는 ]가 대체된 새로운 파일 경로 얻기
    new_path = replace_brackets_with_underscore_file_name(original_path)
    
    # 파일 이름 변경
    if original_path != new_path:
        
        try:
            
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            shutil.move(original_path, new_path)
            info_message = ( f"파일 이름 변경 성공: {original_path} -> {new_path}")
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            counters.increment_rename_count()
            return new_path
        
        except OSError as e:
            info_message = (f"파일 이름 변경 실패 - 00: {original_path} -> {new_path}\n에러 메시지: {e}")
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
            # log_info.status_info_print( info_message)  
            try:
                
                shutil.move(original_path, new_path)
                info_message = (f"파일 이름 변경 성공 - 01: {original_path} -> {new_path}")
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
                counters.increment_rename_count()
                return new_path
            
            except OSError as e:
 
                info_message = (f"파일 이름 변경 실패 01: {original_path} -> {new_path}\n에러 메시지: {e}")
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                return None
    else:
        return new_path
    
def get_folder_size_with_list(folder_path, filelist):
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            filelist.append(filepath)
            # total_size += os.path.getsize(filepath)
            try:
                total_size += os.path.getsize(f"""{filepath}""")
            except Exception as e:
                log_info.status_info_print( f"get size : {folder_path} :  {str(e)} ")
            

    return total_size

def add_basic_meta_info(metainfo, para_1, para_2, para_3, para_4, para_5, para_6 ):

    # file_meta = {
        
    #     "@timestamp": "",
    #     "file" : {
    #         "accessed" : 접근 시간,
    #         "created" : 생성 시간,
    #         "ctime" : 메타 변경 시간,
    #         "mtime" : 마지막 수정 시간,
    #         "directory" : 파일 경로,
    #         "name" : 파일 이름,
    #         "path" : 파일 전체 경로,
    #         "message" : 파일 내용,
    #         "summary" : 파일 요약(300자),
    #         "attributes" : 속성(ex. archive, compressed, directory, encrypted, execute, hidden, read, readoly, system, write, etc.. ),
    #         "owner" : 파일 owner 이름,
    #         "extension" : 확장자,
    #         "mime_type" : 컨텐츠 타입(ex. image/png),
    #         "size" : 파일 사이즈,
    #         "type" : 파일 유형(file, dir, symlink, etc.. ),
    #         "meta_info" : {
            
    #         }
    #     },
    #     "uuid": ' ',
    #     "tags" : ["file", "email"] 
    # }   
     
     
    
    metainfo["meta_info"] = """{}""".format(para_1)
    metainfo["accessed"] = """{}""".format(para_2)
    metainfo["ctime"] = """{}""".format(para_3)
    metainfo["mtime"] = """{}""".format(para_4)
    metainfo["size"] = """{}""".format(para_5)
    metainfo["owner"] = """{}""".format(para_6)  
    log_info.debug_print(metainfo)
    
    return metainfo  