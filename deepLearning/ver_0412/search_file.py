# -*- coding: utf-8 -*-

import os
import time
import json

from bs4 import BeautifulSoup
#import html2text

# from tika import parser
import tika
from tika import parser
import re
import requests
import requests.exceptions
import threading
import concurrent.futures
from elasticsearch.exceptions import TransportError, AuthenticationException
from elasticsearch import Elasticsearch
import datetime
import uuid

from concurrent.futures import ThreadPoolExecutor

import asyncio

import shutil
import subprocess

import config_indexing as cf_index

import file_save_mng as file_es 


import xml.etree.ElementTree as ET


from meta_info import CountersWithLock, MetaInfo # 모듈에서 클래스를 가져옴

import file_save_mng as file_es 

#es connection and indexing
import es_mng 

import psutil
import logging  # logging 모듈을 import
import log_info  # log_info 모듈을 임포트하여 설정을 공유
import sys
import socket
import magic

import mail_utility as mail_ut
import main_utility as main_ut

from maillist import eml_files as eml
from maillist import mbox_files as mbox
from maillist import edb_files as edb
from maillist import nsf_files as nsf

from maillist import msg_files as msg
from maillist import pst_files_libpff as pst

from decompress import recover_extract as decompress_file
from crypto_detection import encrypted_file as encrypt_ut
import ocr_parser as ocr_mag

# from meta_info import doc_meta_info 
# from meta_info import mail_meta_info 

# 락 객체 생성
# doc_lock = threading.Lock()
# config_lock = threading.Lock()
tika_server_lock = threading.Lock()
 
# main.py

gstart_time = 0
gend_time = 0 
gcopy_time = 0
gsecond_time = 0
gdelete_time = 0


# CPU 사용률 임계값 (0.8은 80%를 나타냅니다)

def kill_tika_by_port(main_fold, port):

    try:
        command = f"pkill -f 'tika-server.jar --port={port}'"
        subprocess.run(command, shell=True)
        log_info.status_info_print(f"Killing tika : {main_fold} - {command}")
        time.sleep(5)
        
    except  Exception as e:
        info_message = f"{main_fold} : exceptions - port : {port}: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
 

def get_file_signature(file_path, num_bytes=4):
    with open(file_path, 'rb') as f:
        signature = f.read(num_bytes)
    return signature.hex().upper()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_tika_server(port_number, java_command):
    current_directory = os.getcwd()
    tika_server_path = os.path.join(current_directory, "tika-server.jar")
    if not is_port_in_use(port_number):
        
        # java -Xms512m -Xmx1024m -jar tika-server.jar

        # tika_command = f'java -jar {tika_server_path} --port={port_number} --host localhost &'
        # tika_command = f'nohup java -Xms512m -Xmx1024m -jar "{tika_server_path}" --port={port_number} --host localhost > /dev/null 2>&1 &'
      
        tika_command = f'{java_command} "{tika_server_path}" --port={port_number} --host localhost > /dev/null 2>&1 &'
        try:
            subprocess.run(tika_command, shell=True)
            log_info.status_info_print(f"start tika server : {tika_command}")
        except Exception as e:
            info_message = f"Exception : {str(e)}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)     
        time.sleep(2)  # 5초간 대기
        

def get_file_count(directory_path, filterinfolist, counters):
    try:
        compression_extensions  = filterinfolist[1]
        with os.scandir(directory_path) as entries:

            for entry in entries:
                if entry.is_file():
                    # os.chmod(entry.path, 0o700)
                    counters.increment_files_count()
                    if not os.path.islink(entry.path):
                        try:  
                            file_ext = os.path.splitext(entry.path)[1].lower() 
                            # log_info.status_info_print( f"  get_file_count :  {entry.path} : {file_ext}")
                            if file_ext in compression_extensions:
                                info = main_ut.make_error_data(entry.path, "압축파일")
                                counters.add_compress_file_list(info)                           
                        except Exception as e:
                            info_message = f"Exception 0: {str(e)}"
                            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)  
                        
                elif entry.is_dir():
                    # os.chmod(entry.path, 0o700)
                    get_file_count(entry.path, filterinfolist, counters)
            

    except Exception as e:
        info_message = f"Exception 1: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
              
  
  
def get_file_names(directory_path, total_list, filterinfolist, counters, ocrflag):
    try:       
        file_filter = filterinfolist[0]
        compression_extensions  = filterinfolist[1]
        imageinfolist = filterinfolist[5] 
    
        checker = main_ut.FileEncryptChecker()  
        
        if not os.path.exists(directory_path):
            info_message = f" : {entry.path}, error:  there is no directory_path"
            log_info.debug_print(info_message) 
            return False
        # log_info.status_info_print( f"  file_filter :  {file_filter} ")
        with os.scandir(directory_path) as entries:
            for entry in entries:
                try:                
                
                    if entry.is_file():

                        counters.increment_files_count()
                        # os.chmod(entry.path, 0o700)
                        if not os.path.islink(entry.path):
                            try:
                                file_size = os.path.getsize(entry.path) 
                                # log_info.status_info_print( f"  file_size :  {file_size} ") 
                                if file_size == 0:
                                    try:
                                        info = main_ut.make_error_data(entry.path,  "size zero")
                                        counters.add_analyzer_issue_list(info)
                                        counters.increment_analyzer_issue_list_count()
                                    except OSError as e:
                                        info_message = f" : {entry.path}, zeor size file of OSError so moving error: {str(e)}"
                                        log_info.debug_print(log_info.logger, info_message, log_level=logging.INFO) 
                                        
                                    except Exception as e:
                                        counters.increment_files_without_extensions() 
                                        info_message = f"Exception 0: {entry.path} - {str(e)}"
                                        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)                            
                                                            
                                else:
                    
                                    file_ext = os.path.splitext(entry.path)[1].lower() 
                                    # log_info.status_info_print( f"  file_ext :  {file_ext} ")
                                    if file_ext in file_filter:  
                                        
                                        if ocrflag is False:
                                            if file_ext in imageinfolist: 
                                                
                                                info = main_ut.make_error_data(f"""{entry.path}""",  "분석제외파일")
                                                info["file_ext"] = str(file_ext)
                                                counters.add_file_ext_list(info)
                                                counters.increment_ext_list_count()
                                                continue
                                        
                                        if True: #WHITE_MODE:
                                            
                                            if main_ut.identify_file_type_new(entry.path, file_ext, counters):#signiture check 
                                                counters.increment_fileinfo_analyze_count()
                                                new_path = main_ut.rename_file_with_underscore(entry.path, counters)
                
                                                if new_path is None:
                                                    continue
                                                    # log_info.status_info_print( f"  6 :  {entry.path} : {new_path}")

                                                    # log_info.status_info_print( f"  7 :  {entry.path} : {new_path}")
                                                    
                                                
                                                if file_ext in compression_extensions:
                                                    info = main_ut.make_error_data(new_path,  "압축파일")
                                                    counters.add_compress_file_list(info)
                                                    
                                                checker = main_ut.drm_check_file(new_path)   
                                                
                                                if checker is not None:
                                                    counters.add_drm_file_list(checker)
                                                    continue
                                                
                                                is_encrypted, algorithm = encrypt_ut.is_encrypted_file(new_path)
                                                # log_info.status_info_print( f"  6 :  {entry.path} : {is_encrypted}")
                                                if is_encrypted:
                                                    # counters.add_cry_file_list(new_path)
                                                    info = main_ut.make_error_data(entry.path,  "암호화 파일")
                                                    counters.add_file_issue_list(info)
                                                    counters.increment_file_issue_list_count()
                                                    continue
                                                total_list.append(new_path)                                                        
                                            else:                    
                                                info = main_ut.make_error_data(entry.path,  "wrong signiture")
                                                counters.add_file_issue_list(info)
                                                counters.increment_file_issue_list_count()

                                            
                                    else:
                                        info = main_ut.make_error_data(f"""{entry.path}""",  "분석제외파일")
                                        info["file_ext"] = str(file_ext)
                                        counters.add_file_ext_list(info)
                                        counters.increment_ext_list_count()
                                        
                            except Exception as e:
                                counters.increment_files_without_extensions() 
                                info_message = f"Exception 0: {entry.path} - {str(e)}"
                                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)                            
                                
                        else:
                            counters.increment_files_symbolic_count()
                            info = main_ut.make_error_data(entry.path,  "symbolic file")
                            counters.add_file_issue_list(info)
                            counters.increment_file_issue_list_count()
                            
                    elif entry.is_dir():
                        # os.chmod(entry.path, 0o700)
                        get_file_names(entry.path, total_list, filterinfolist, counters, ocrflag)
            

                except Exception as e:

                    info_message = f"Exception 1: {str(e)}"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                    info = main_ut.make_error_data(entry.path,  "no reason")
                    counters.add_file_issue_list(info)
                    counters.increment_file_issue_list_count()
            
        return True
    except Exception as e:
        counters.increment_files_without_extensions() 
        info_message = f"Exception 0000: {entry.path} - {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)        
def list_sorting(total_list, sortingenable, counters):
    # 파일 크기를 기준으로 내림차순으로 정렬

   
    if sortingenable:
       
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_info.status_info_print( f"\n list_sorting {len(total_list)} - sorting start time : {current_time}")

        start_time = time.time()
        total_list.sort(key=lambda x: os.path.getsize(x), reverse=False)   
        end_time = time.time()
        log_info.status_info_print( f"\n list_sorting {len(total_list)}- sorting file names Time: %.2f seconds" % (end_time - start_time))

        
    return total_list
             
def tika_process_memory(tika_jar_path, cncoding_type, data):
    tika_command = f'java -jar "{tika_jar_path}" -'
    result = subprocess.run(tika_command, shell=True, check=True, encoding=cncoding_type, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout


def tika_process(tika_jar_path, option_mode,  cncoding_type, file_path):
    
    abs_document_file_path = os.path.abspath(file_path)
    tika_command = f' java -jar "{tika_jar_path}" {option_mode} "{abs_document_file_path}"'

    log_info.log_with_function_name(log_info.logger, tika_command, log_level=logging.DEBUG) 
    result = subprocess.run(tika_command, shell=True, check=True, encoding=cncoding_type, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # time.sleep(0.001)
    return result.stdout

def process_tika_app_thread(tika_jar_path, tika_jar_name, file_path, output_directory, option_mode, cncoding_type, issue_path, filesize, counters):
    
    start_time = time.time()  
    # file_path = "/data/data/test_1G/'test path_01'/웹POI추출_v1.4.xlsx"
    
    abs_document_file_path = os.path.abspath(file_path)

    current_directory = os.getcwd()
    # print("현재 디렉토리:", current_directory)
    
    if tika_jar_path is None or not tika_jar_path.strip():
        tika_jar_path  = os.path.join(current_directory, tika_jar_name )
    try:
        # tika_command =  f"java -jar /home/nettars/BMT/k8096_mhpark/pre_processor/tika-app-2.8.0.jar -e '/data/data/D2/2. 작업내용 및 문서/2. 3S소프트/DELL서버/기타업데이트/QLogic_QC_E4_Manual_External_End_User_35.30.00.09/SimpChin/UsersGuide_FCAdapter_27xx-DEL_BK3254601-05N.pdf'"
        # print(f"Tika app cmd : ", tika_command)
        
        futures = []
        optionlist = {'-m', '-t'}
        cncoding_type = "utf-8"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for option_mode in optionlist:

                future = executor.submit(tika_process, f"{tika_jar_path}", option_mode,  cncoding_type, file_path)
                futures.append(future)
                time.sleep(0.001)
                
            concurrent.futures.wait(futures)
            executor.shutdown()


        # 쓰레드 풀 종료
        # executor.shutdown()
        
        metadata_info = None  
        text = None
        
        for future in futures:
            
            result = future.result()
              
            if result is None:
                continue  # 결과가 없을 경우 다음 작업으로 건너뛰기
            try:
                if "Content-Length:" in result or "Content-Type:" in result:
                    metadata_info = main_ut.from_string_parse_json(result)
                else:
                    text = result
                    # log_info.status_info_print( "tika command end 4" , result)
                
            except KeyError:
                content_length = 0
            
            # info_message = f"{file_path}, size({filesize}) Tika parsing body time (T-{(end_time - start_time)})"
            # log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
                  
        end_time = time.time()  
        info_message = f"{file_path}, size({filesize}) : Tika parsing time (T-{(end_time - start_time)})"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
        return metadata_info, text      
                                
    except subprocess.CalledProcessError as e:
       
        # Tika 애플리케이션이 오류를 반환할 때 암호화 예외 처리를 수행
        if "Unable to process: document is encrypted" in str(e):
            try:

                # shutil.move(file_path, issue_path)
                info = main_ut.make_error_data(file_path, "암호화 파일")
                counters.add_analyzer_issue_list(info)
                counters.increment_analyzer_issue_list_count()
                # 암호화된 PDF 파일을 처리하는 특별한 로직을 추가하세요
                info_message = f"이 파일은 암호화되어 있습니다. 별도의 처리가 필요합니다.: {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            except Exception as e:
                info_message = f"Move File : {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                # return None
            
        elif "returned non-zero exit status 1" in str(e):
            try:
                # shutil.move(file_path, issue_path)
                
                info = main_ut.make_error_data(file_path,  "non-zero")
                counters.add_analyzer_issue_list(info)
                counters.increment_analyzer_issue_list_count()
                # 암호화된 PDF 파일을 처리하는 특별한 로직을 추가하세요
                info_message = f"이 예외가 있습니다. 별도의 처리가 필요합니다.: {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            except Exception as e:
                info_message = f"Move File : {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                # return None         
        else:
            # 다른 예외 처리를 수행
            info_message = f"TiKa Exception 0 subprocess.CalledProcessError: {file_path}: {str(e)}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
        
        # return None
  
    except Exception as e:
        info_message = f"Exception 1: {file_path}: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)  
        
    end_time = time.time() 
    info_message = f"{file_path}, Tika parsing all exception time (T-{(end_time - start_time)})"
    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)       
    return None, None


def process_tika_app(tika_jar_path, tika_jar_name, file_path, output_directory, option_mode, cncoding_type, issue_path, filesize, counters):
    # file_path = "/data/data/test_1G/'test path_01'/웹POI추출_v1.4.xlsx"
    log_info.debug_print( f" start tika-app : {file_path} ")
    start_time = time.time()  
    # abs_document_file_path = os.path.abspath(file_path)

    current_directory = os.getcwd()
    # print("현재 디렉토리:", current_directory)
    
    if tika_jar_path is None or not tika_jar_path.strip():
        
        tika_jar_path  = os.path.join(current_directory, tika_jar_name )
        
    try:
        # tika_command =  f"java -jar /home/nettars/BMT/k8096_mhpark/pre_processor/tika-app-2.8.0.jar -e '/data/data/D2/2. 작업내용 및 문서/2. 3S소프트/DELL서버/기타업데이트/QLogic_QC_E4_Manual_External_End_User_35.30.00.09/SimpChin/UsersGuide_FCAdapter_27xx-DEL_BK3254601-05N.pdf'"
        # print(f"Tika app cmd : ", tika_command)
        tika_command = f'java -jar "{tika_jar_path}" {option_mode} "{file_path}"'
             
        if option_mode == '-m':
            # tika_command = f"java -jar {tika_jar_path} {option_mode} '{abs_document_file_path}'"
            # Tika JAR 실행
            result = subprocess.run(tika_command, shell=True, check=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,)
            end_time = time.time() 
            info_message = f"{file_path}, size({filesize})kb Tika parsing meta time (T-{(end_time - start_time)})"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
            return main_ut.from_string_parse_json(result.stdout)          
        else: 
            # tika_command = f"java -jar {tika_jar_path} {option_mode} '{abs_document_file_path}'"
            # Tika JAR 실행
            # log_info.debug_print( "start body : ", source_path)
            result = subprocess.run(tika_command, shell=True, check=True, encoding=cncoding_type, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            end_time = time.time() 
            info_message = f"{file_path}, size({filesize})kb Tika parsing body time (T-{(end_time - start_time)})"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
            return result.stdout  
                                   
    except subprocess.CalledProcessError as e:
       
        # Tika 애플리케이션이 오류를 반환할 때 암호화 예외 처리를 수행
        if "Unable to process: document is encrypted" in str(e):
            try:

                # shutil.move(file_path, issue_path)
                info = main_ut.make_error_data(file_path,  "암호화 파일")
                counters.add_analyzer_issue_list(info)
                counters.increment_analyzer_issue_list_count()
                # 암호화된 PDF 파일을 처리하는 특별한 로직을 추가하세요
                info_message = f"이 파일은 암호화되어 있습니다. 별도의 처리가 필요합니다.: {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            except Exception as e:
                info_message = f"Move File : {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
                # return None
            
        elif "returned non-zero exit status 1" in str(e):
            try:
                # shutil.move(file_path, issue_path)
                info = main_ut.make_error_data(file_path,  "non-zero")
                counters.add_analyzer_issue_list(info)
                counters.increment_analyzer_issue_list_count()
                # 암호화된 PDF 파일을 처리하는 특별한 로직을 추가하세요
                info_message = f"이 예외가 있습니다. 별도의 처리가 필요합니다.: {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            except Exception as e:
                info_message = f"Move File : {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
                # return None         
        else:
            # 다른 예외 처리를 수행
            info_message = f"TiKa Exception 0 subprocess.CalledProcessError: {file_path}: {str(e)}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
        
        # return None
  
    except Exception as e:
        info_message = f"TiKa Exception 1: {file_path}: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)  
        
    end_time = time.time() 
    info_message = f"{file_path}, Tika parsing all exception time (T-{(end_time - start_time)})"
    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)       
    return None

def check_tika_server_response(file_path, tika_server_endpoint, counters):

    try:
        response = requests.get(tika_server_endpoint)
        if response.status_code == 200:
            
            info_message = f"Tika 서버가 정상적으로 작동 중입니다. t_c : {counters.tika_count}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            return True
        
        else:
            info_message = f"Tika 서버 응답 상태 코드 : {file_path} :  {response.status_code} : t_c : {counters.tika_count}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 

        
    except requests.ConnectionError:
        info_message = f"Tika 서버에 연결할 수 없습니다. {file_path} : {tika_server_endpoint} : {counters.tika_count}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 

        
    except Exception as e:
        # if try_count >= 3:
        end_time = time.time()
        info_message = f"{file_path} : {counters.tika_count} Execption - tika-error - {str(e)})"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)

    return False
       
def process_tika_server(tika_server_endpoint, tika_jar_name, file_path, output_directory, option_mode, cncoding_type, issue_path, filesize, counters):
    
    # file_path = "/data/data/test_1G/'test path_01'/웹POI추출_v1.4.xlsx"
    file_path = f"""{file_path}"""
    log_info.debug_print( f" start tika-server : {file_path} ")
    start_time = time.time()  
    # abs_document_file_path = os.path.abspath(file_path)

    current_directory = os.getcwd()
    # print("현재 디렉토리:", current_directory)
    step = 0
    try:
        # file_path =  "/data/data/storage2/02. 서버기술팀/2. 작업내용 및 문서/2018년/201805 고등과학원/★10. 고등과학원 최종제출/ppt/발표자료/제안발표자료_v1.0_20180726_최종.pptx"
                    # log_info.status_info_print(meta_data)
        # file_path = "/data/10_t/temp_fold/xlsx4 - 복사본 (4).xlsx"
        # file_path = "/data/10_t/temp_fold/w16_film_festival_poster___복사본__15____복사본.docx" 
        # curl -T "/data/source_data/eml/100G/6/서울지방국세청___복사본/old/국세청_포렌식_고도화_VMware___복사본__2_.pptx" http://127.0.0.1:10019/tika

        max_retries = 2
        try_count = 0

        parsed = None
        
        while try_count < max_retries:
            try_count += 1
            
            if check_tika_server_response(file_path, tika_server_endpoint, counters):
                
                try:
                    counters.increment_tika_count()
                    parsed = parser.from_file(f"""{file_path}""", serverEndpoint=tika_server_endpoint, requestOptions={'timeout': 240})
                    counters.decrement_tika_count()
                    break  # 정상적으로 응답이 왔을 경우 루프 종료

                except TimeoutError as te:
                    counters.increment_tika_timeout_count()
                    if try_count >  max_retries:
                        end_time = time.time()
                        info_message = f"{file_path} : {counters.tika_count} Timeout Error (T-{(end_time - start_time)}): {str(te)}"
                        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
                        break

                except Exception as e:
                    end_time = time.time()
                    info_message = f"{file_path} : {counters.tika_count} Execption - tika-error (T-{(end_time - start_time)} - {str(e)})"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
             
            else:
                time.sleep(1)       
            
        if parsed is not None: 
                
            status = parsed.get('status', None)
            
            if status == 200:
                try:
                    body_info = parsed.get('content', None)
                    metainfo = parsed.get('metadata', None)
                    end_time = time.time() 
                    info_message = f"{file_path}, t_c : {counters.tika_count} size({filesize})kb Tika parsing meta and body time (T-{(end_time - start_time)})"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
                    # log_info.status_info_print( f" {file_path} : tika-info : {metainfo} - {body_info}")
                    return metainfo, body_info 
                except Exception as e:

                    info_message = f"{file_path} : {counters.tika_count} Execption -parsed- {str(e)})"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)        
        return None, None
                                   
    except subprocess.CalledProcessError as e:
        # Tika 애플리케이션이 오류를 반환할 때 암호화 예외 처리를 수행
        if "Unable to process: document is encrypted" in str(e):
            try:

                # shutil.move(file_path, issue_path)
                info = main_ut.make_error_data(file_path,  "암호화 파일")
                counters.add_analyzer_issue_list(info)
                counters.increment_analyzer_issue_list_count()
                # 암호화된 PDF 파일을 처리하는 특별한 로직을 추가하세요
                info_message = f"이 파일은 암호화되어 있습니다. 별도의 처리가 필요합니다.: {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            except Exception as e:
                info_message = f"Move File : {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 

            
        elif "returned non-zero exit status 1" in str(e):
            try:
                # shutil.move(file_path, issue_path)
                info = main_ut.make_error_data(file_path,  "non-zero")
                counters.add_analyzer_issue_list(info)
                counters.increment_analyzer_issue_list_count()
                # 암호화된 PDF 파일을 처리하는 특별한 로직을 추가하세요
                info_message = f"이 예외가 있습니다. 별도의 처리가 필요합니다.: {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
            except Exception as e:
                info_message = f"Move File : {file_path}: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                # return None         
        else:
            # 다른 예외 처리를 수행
            info_message = f"TiKa Exception 0 subprocess.CalledProcessError: {file_path}: {str(e)}"
            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
        
        # return None
  
    except Exception as e:
        end_time = time.time() 
        info_message = f" Exception 1:{file_path}: t_c : {counters.tika_count} : serverEndpoint={tika_server_endpoint}: (T-{(end_time - start_time)} :{str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
        log_info.debug_print(info_message) 
        
        # time.sleep(0.500)
               
    end_time = time.time() 
    info_message = f"{file_path}, Tika parsing all exception time (T-{(end_time - start_time)})"
    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)       
    return None, None

import xml.etree.ElementTree as ET  
 


def fileinfo_analyze_app(sindex, allcount, activate_thread_count, file_path, path_info, es, jsondata, file_type, tika_server_mode, filterinfolist, counters, target_path, issue_path, main_fold, sub_main_fold, edbflag, nsfflag, ocrflag):


    try:
        analyzerinfolist = filterinfolist[2]
        file_read_extensions = filterinfolist[3]
        email_extensions = filterinfolist[4]
        imageinfolist = filterinfolist[5]    
        
        counters.add_debugging_file_list(sindex, f"""{sindex} - {file_path}""")
        
        now = datetime.datetime.now()
                    
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

        # mime = magic.Magic(mime=True)
        mime_type = file_ext[1:]
        docmetainfo = {}
        metadata_info = None    
        metadata_info, text = main_ut.read_file_from_path(file_path, counters, False)
        
        if metadata_info is None:
            info = main_ut.make_error_data(file_path,  "fail to get file metainfo")
            counters.add_analyzer_issue_list(info)
            counters.increment_analyzer_issue_list_count()
            # log_info.status_info_print(info)  
            # counters.del_debugging_file_list(sindex) 
            return
        else:
            mime_type = metadata_info.mime_type

    except Exception as e:
        info_message = f" : {file_path}, Exception read_file_from_path: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
    # log_info.status_info_print(f" ***** fileinfo_analyze_app **** : {file_ext} : {edbflag}: {nsfflag}: {ocrflag}")

    try:           

        text = None 
        step_num = 99
        docmetainfo["tags"] = []
        docmetainfo["date"] = metadata_info['created']
        if main_ut.check_signiture_check(file_ext[1:], mime_type) and (file_ext[1:] != mime_type):
            docmetainfo["tags"].append("sig")
                                          
        if file_ext in analyzerinfolist:
            step_num = 77
            docmetainfo["file"] = {}
            try:

                docmetainfo["root_path"] = main_fold 
                docmetainfo["directory"] = splitdrive_file_path
                docmetainfo["title"] = file_name
                docmetainfo["file"]["path"] = file_path
                docmetainfo["file"]["extension"] = file_ext[1:]
                docmetainfo["file"]["size"] = file_size
                docmetainfo["uuid"] = uuid_str 
                # log_info.process_status_info_print(f"aaaa start : {file_type} - {main_fold} - {sindex} - {allcount} : {file_path}")
                docmetainfo["tags"].append("file")
                docmetainfo["tags"].append(f"{file_type}")


                docmetainfo["@timestamp"] = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                # 파일 처리
                docmetainfo["file"]["type"] = "file"
                docmetainfo["file"]["mime_type"] = f"file/{mime_type}"
            except Exception as e:
                info_message = f" : {file_path}, Exception 0: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 

            # 파일 처리
            try:
                # file_path =  "/data/data/D2/2. 작업내용 및 문서/2. 3S소프트/DELL서버/기타업데이트/QLogic_QC_E4_Manual_External_End_User_35.30.00.09/SimpChin/UsersGuide_FCAdapter_27xx-DEL_BK3254601-05N.pdf"
                # file_path = "/data/upload/upload/target_data/부인상사_left-com__82108/_del_/D∶/_Lost Files/DEL_14_1_워드 문서 열람 리스트.csv"

                if file_size == 0:
                    counters.increment_files_exception_count()
                    log_info.debug_print( f" {file_path} size is {file_size}")

                    try:

                        info = main_ut.make_error_data(file_path,  "size zero")
                        counters.add_analyzer_issue_list(info)
                        counters.increment_analyzer_issue_list_count()
                    except OSError as e:
                        info_message = f" : {file_path}, zeor size file of OSError so moving error: {str(e)}"
                        log_info.debug_print(log_info.logger, info_message, log_level=logging.INFO) 
                    return
                else:
                    if file_ext in file_read_extensions:

                        try:

                            metadata_info, text = main_ut.read_file_from_path(file_path, counters, True)

                        except Exception as e:
                            info_message = f" : {file_path}, Exception read_file_from_path: {str(e)}"
                            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO) 
                            
                        if metadata_info is None:
                            info = main_ut.make_error_data(file_path,  "fail to get file metainfo")
                            counters.add_analyzer_issue_list(info)
                            counters.increment_analyzer_issue_list_count()
                            return
                        else:
                            step_num = 1    
                            try:
                                step_num = 2
                                docmetainfo["file"]["accessed"] = metadata_info["accessed"]
                            except KeyError:
                                step_num = 3
                                pass
                            step_num = 4  
                            try:
                                # docmetainfo["file"]["created"] = metadata_info["created"]  
                                docmetainfo["date"] = metadata_info["created"]  
                            except KeyError:
                                pass
                            # try:
                            #     docmetainfo["file"]["ctime"] = metadata_info["ctime"]
                            # except KeyError:
                            #     pass
                            step_num = 5  
                            try:
                                docmetainfo["file"]["mtime"] = metadata_info["mtime"]
                            except KeyError:
                                pass
                            step_num = 6   
                            try:
                                docmetainfo["file"]["owner"] = metadata_info["owner"]
                            except KeyError:
                                pass
                            step_num = 7  
                            try:
                                docmetainfo["file"]["size"] = metadata_info["size"]
                            except KeyError:
                                pass 
                            step_num = 8       
                            docmetainfo["file"]["meta_info"] = f"""{metadata_info}""" 
                            step_num = 9
                                             
                    else:
                        step_num = 20
                        
                        if file_size >= (1024 * 1024 * 1024):
                            counters.increment_files_exception_count()
                            info_message = f" : {file_path}, this is too big size to call tika server: {file_size/(1024 * 1024)}Mbyte"
                            log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 

                            info = main_ut.make_error_data(file_path,  "too big for parser")
                            counters.add_analyzer_issue_list(info)
                            counters.increment_analyzer_issue_list_count()
                            return 
                        
                        else:
                            try:

                                if tika_server_mode:
                                    metadata_info, text =  process_tika_server(jsondata["tika_app"]["tika_server_endpoint"], jsondata["tika_app"]["tika_app_jar_name"], file_path, "", '-m', "", issue_path, file_size, counters) 
                                else:
                                    metadata_info, text =  process_tika_app_thread(jsondata["tika_app"]["tika_app_jar_path"], jsondata["tika_app"]["tika_app_jar_name"], file_path, "", '-m', "", issue_path, file_size, counters) 

                            except Exception as e:
                                info_message = f" : {file_path}, Exception tika_server_mode or thread: {str(e)}"
                                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                                
                        step_num = 1211    
                        if metadata_info is None:
                            step_num = 22
                            counters.increment_files_exception_count()
                        else:
                            step_num = 2210
                            
                            if isinstance(metadata_info, str):
                                metadata_info = json.loads(metadata_info)
                                
                            try: 
                                step_num = 2212
                                docmetainfo["file"]["type"] = metadata_info["Content-Type"]
                                step_num = 2210
                            except Exception as e:
                                info_message = f" : {file_path}, Exception Content-Type:  - {str(e)}"
                                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                                pass
                            step_num = 2215
                            try:
                                docmetainfo["date"] = metadata_info['dcterms:created']  
                                # docmetainfo["file"]["created"] = metadata_info['dcterms:created'] 
                            except Exception as e:
                                pass
                            step_num = 2216
                            try:
                                docmetainfo["file"]["mime_type"] = metadata_info["Content-Type"]
                                    
                            except Exception as e:
                                pass
                            step_num = 2217
                            try:
                                docmetainfo["file"]["mtime"] = metadata_info["dcterms:modified"]
                            except Exception as e:
                                pass
                            step_num = 2218
                            try:
                                docmetainfo["file"]["owner"] = metadata_info["meta:last-author"]
                            except Exception as e:
                                pass
                            step_num = 2219
                            try:
                                docmetainfo["file"]["size"] = metadata_info["Content-Length"]
                            except Exception:
                                pass  
                            step_num = 30 
                            
                            try:
                                Content_Encoding = metadata_info["Content-Encoding"]  
                            except Exception as e:
                                Content_Encoding = None
                                pass
                     
                            try:  
                                docmetainfo["file"]["meta_info"] = f"""{metadata_info}""" 
                            except Exception as e:
                                pass    
                            step_num = 32
                            
                            if tika_server_mode:
                                step_num = 201
                                pass
                            else:    
                                step_num = 211
                                if Content_Encoding is None:
                                    pass
                                    # Content_Encoding = "utf-8" 
                                    # with doc_lock:  
                                    #     tika_count_start += 1 
                                    # step_num = 212
                                    # info_message = f" : {file_path}, text : {text}"
                                    # log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR) 
                                    # # text = process_tika_app(jsondata["tika_app"]["tika_app_jar_path"], jsondata["tika_app"]["tika_app_jar_name"], file_path, "", '-t', Content_Encoding, path_info["issue_path"], file_size)
                                    # with doc_lock:  
                                    #     tika_count_end += 1
                                    #     tika_count = tika_count_end - tika_count_start   
                                    # time.sleep(0.001)
                    step_num = 23
                    sleep_time = jsondata['sleep_time']
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                    if text is None:
                        step_num = 24
                        pass
                    else:
                        show_content, text = mail_ut.truncate_string(text) 

                        if show_content is not None:   
                            docmetainfo["summary"] = """{}""".format(show_content)
                        # docmetainfo["file"]["summary"] = """{}""".format(show_content)
                        # docmetainfo["date"] = created_date
                        if text is not None:
                            docmetainfo["content"] = text
                        # docmetainfo["file"]["message"] = text
                            
            except requests.exceptions.RequestException as e:
                info_message = f"{file_path}, requests.exceptions.RequestException: {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)

                counters.increment_files_exception_count()

            except Exception as e:      
                info_message = f"{file_path}, exceptions 11 - {step_num} : {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
                counters.increment_files_exception_count()
            
                
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
                            # sys.exit(1)  

        elif file_ext in email_extensions or (edbflag  and file_ext == ".edb" ) or (nsfflag  and file_ext == ".nsf" ):   

            # log_info.status_info_print( f" start mail {file_path} ******************** {file_ext}\n\n") 
            step_num = 100   
            docmetainfo["email"] = {}
            # docmetainfo["email"]["origin"] = {}
            try:
                docmetainfo["root_path"] = main_fold 
                docmetainfo["email"]["path"] = file_path
                docmetainfo["directory"] = splitdrive_file_path 
                docmetainfo["email"]["extension"] = file_ext[1:]                
                docmetainfo["email"]["name"] = file_name

                # docmetainfo["email"]["extension"] = file_ext[1:]
                # docmetainfo["email"]["size"] = file_size
                docmetainfo["uuid"] = uuid_str        
                docmetainfo["@timestamp"] = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')   
                # docmetainfo["tags"] = ["mail",f"{file_type}"]
                docmetainfo["tags"].append("mail")
                docmetainfo["tags"].append(f"{file_type}")                                
            except Exception as e:
                info_message = f"{file_path}, exceptions : {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)

            if file_ext == '.pst':
                # counters.increment_fileinfo_analyze_count() 

                docmetainfo = pst.read_pst_file(file_path, path_info["pst_attachment_path"], target_path, es, jsondata, docmetainfo, counters)
                counters.increment_pst_count() 

            elif file_ext == '.eml':
                # counters.increment_fileinfo_analyze_count() 
                docmetainfo = eml.read_eml_file(file_path, path_info["eml_attachment_path"], target_path, es, jsondata, docmetainfo, counters)
                counters.increment_eml_count() 

            elif file_ext == '.msg':
                # counters.increment_fileinfo_analyze_count() 
                docmetainfo = msg.read_extract_msg_info(file_path, path_info["msg_attachment_path"], target_path, es, jsondata, docmetainfo, counters)
                counters.increment_msg_count() 
            elif file_ext == '.mbox':
                # counters.increment_fileinfo_analyze_count() 
                try:

                    docmetainfo = mbox.read_mbox_file(file_path, path_info["mbox_attachment_path"], target_path, es, jsondata, docmetainfo, counters)
                    counters.increment_mbox_count()  
                except Exception as e:
                    info_message = f" : {file_path}, Exception mbox: {str(e)}"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)                 

            elif file_ext == '.edb':
                docmetainfo = edb.read_edb_file(file_path, path_info["edb_attachment_path"], target_path, es, jsondata, docmetainfo, counters)
                counters.increment_edb_count() 

            elif file_ext == '.nsf':
                docmetainfo = nsf.read_nsf_file(file_path, path_info["nsf_attachment_path"], target_path, es, jsondata, docmetainfo, counters)
                counters.increment_nsf_count() 

        else: 
            if ocrflag and (file_ext in imageinfolist):
                # log_info.status_info_print(f"{ocrflag}  {file_ext}: {file_path}")  
                counters.increment_files_exception_count()
                    
            else:                    
                docmetainfo["file"] = {}
                try:

                    docmetainfo["root_path"] = main_fold 
                    docmetainfo["directory"] = splitdrive_file_path
                    docmetainfo["title"] = file_name
                    docmetainfo["file"]["path"] = file_path
                    docmetainfo["file"]["extension"] = file_ext[1:]
                    docmetainfo["file"]["size"] = file_size
                    docmetainfo["uuid"] = uuid_str 
                    # docmetainfo["tags"] = ["file",f"{file_type}"]   
                    docmetainfo["tags"].append("file")
                    docmetainfo["tags"].append(f"{file_type}")       
                    docmetainfo["@timestamp"] = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

                    # 파일 처리
                    docmetainfo["file"]["type"] = "file"
                    docmetainfo["file"]["mime_type"] = f"file/{mime_type}"
            
                    
                except Exception as e:
                    info_message = f" : {file_path}, Exception 0: {str(e)}"
                    log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.INFO)  
    
                step_num = 28
                # log_info.debug_print( f" end fileinfo_analyze :{index} - {allcount} ")

                if es is not None: # indexing document
                    es_status = es_mng.es_index_data(es, jsondata, "el_index_main_name", docmetainfo, file_path, counters)
                    
                    if es_status == False:
                        es = es_mng.es_index_connection(jsondata)
                        if es is None:
                            pass
                            # sys.exit(1)
                        else:
                            es_status = es_mng.es_index_data(es, jsondata, 'el_index_main_name', docmetainfo, file_path, counters)
                            if es_status == False:
                                pass
                                # sys.exit(1)       
    
        # # 2023.08.31
        # source 폴더의 파일 갯수 기준으로 1% 증가 할 때 마다 출력

        counters.increment_processor_count()
        index = counters.processor_count 
        # if sindex == 9:
        #     log_info.status_info_print(f" cc : {sindex} - {file_path}")
        # log_info.process_status_info_print( f" end fileinfo_analyze :{index} - {allcount} ")
        if allcount <= 100 and allcount != 0:   
            step_num = 30
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            activate_thread_count = threading.active_count()  # 현재 활성화된 스레드 수를 가져옵니다.
            log_info.process_status_info_print( f"{current_time} - {main_fold}:{sub_main_fold}-: {index}-{allcount} : thread-N({file_type}):({activate_thread_count}) : processing ... ({index *100/ allcount:.3f}%)")  
        else:     
            step_num = 31
            threshold = max(1, allcount // 100)
            # log_info.process_status_info_print(f"start :{index} - {(index % allcount )} {((index % allcount ) // 1000)} - {main_fold} - {sindex} - {allcount} : {file_path}")
            if index % threshold == 0 or index == allcount:  
                # log_info.process_status_info_print(f"start 1:{index} - {counters.processor_count} {file_type} - {main_fold} - {sindex} - {allcount} : {file_path}")
                activate_thread_count = threading.active_count()  # 현재 활성화된 스레드 수를 가져옵니다.
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_info.process_status_info_print( f"{current_time} - {main_fold}:{sub_main_fold} : {index}-{allcount} : thread-N({file_type}):({activate_thread_count}) : processing ... ({index *100/ allcount:.3f}%)")  
        # time.sleep(0.001)
        
    except Exception as e:      
        info_message = f"{file_path}, exceptions 2 - {step_num }: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
        counters.increment_files_exception_count()  
        
    counters.del_debugging_file_list(sindex)                  
    return None

def process_start(file_list, path_info, es, jsondata, tika_server_mode, file_type, filterinfolist, counters, target_path, issue_path):
    
    # es = None
    try: 

        main_fold = jsondata["main_fold"]
        sub_main_fold = jsondata['sub_main_fold']  
        
        edbflag = jsondata['edbflag']
        nsfflag = jsondata['nsfflag'] 
        ocrflag = jsondata['ocrflag']  
    
        tika_app = jsondata['tika_app']
        tika_server_port = tika_app['tika_server_port']     
   
        start_time = time.time()

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_info.status_info_print( f"\n main_fold : {sub_main_fold} - start Process_start Time : {current_time}")
        info_message = f"tika_server_port 0:{tika_server_port}-{file_type}:{current_time}-{sub_main_fold},  start call file analyzer info - {len(counters.file_path_info)} : {counters.file_path_info} "
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
                
        futures = []
        allcount = len(file_list)
        nindex = 0
        counters.init_debugging_file_list()

        
        with concurrent.futures.ThreadPoolExecutor(max_workers=jsondata["max_thread_num"]) as executor:
            futures = []
            result_list = []
            index = 0
            while index < allcount or futures:
                # 작업 수가 max_workers 이하인 경우에만 작업을 추가합니다.
                # activate_thread_count = threading.active_count()

                if len(futures) < jsondata["max_thread_num"] and index < allcount:
                    # 각 파일을 독립적으로 처리하는 작업을 제출합니다.
                    file_path = file_list[index]
                    # log_info.status_info_print(f"start thread...{index + 1}/{allcount} - {threading.active_count()}")
                    # if "/data/normal/10_t/temp_fold_1_1/20211223_094140__한국전력공사_고객센터___su123kepco_co_kr__요청자료_송부.eml" in file_path:
                    future = executor.submit(fileinfo_analyze_app, (index + 1), allcount, 0, file_path, path_info, es, jsondata, file_type, tika_server_mode, filterinfolist, counters, target_path, issue_path, main_fold, sub_main_fold, edbflag, nsfflag, ocrflag)
                    futures.append(future)
                    index += 1

                else:
                    # 실행 중인 작업 중에서 완료된 작업을 가져옵니다.
                    for future in concurrent.futures.as_completed(futures):
                        # log_info.status_info_print(f"processing result of thread...{index}")
                        result = future.result()
                        # result_count = len(result)
                        # result_list.extend(result)
                        futures.remove(future)
                        # log_info.status_info_print(f"0 finished thread...{index + 1}/{allcount} - {threading.active_count()}")
                        break

                    # time.sleep(0.5)
                    # log_info.status_info_print(f"waitting thread...{index + 1}/{allcount} - {threading.active_count()}")   
                    
            # 나머지 작업을 처리합니다.
            if futures:
                for future in concurrent.futures.as_completed(futures):
                    # log_info.status_info_print(f"processing result of thread...{future}")
                    result = future.result()
                    # result_count = len(result)
                    # result_list.extend(result)
                    log_info.status_info_print(f"1 finished thread...{index + 1}/{allcount} - {threading.active_count()}")

            # 모든 스레드 종료를 기다립니다.
            executor.shutdown(wait=True)

        end_time = time.time() 
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_info.status_info_print( f"{current_time} - {sub_main_fold} : end call file analyzer Time: %.2f seconds" % (end_time - start_time))
        log_info.status_info_print( f"{current_time} - {sub_main_fold} : end call file analyzer info: {len(counters.file_path_info)} : {counters.file_path_info}" )
        info_message = f"tika_server_port 1:{tika_server_port}-{file_type}:{current_time}-{sub_main_fold},  end call file analyzer info - {len(counters.file_path_info)} : {counters.file_path_info} "
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)
    except Exception as e:
        log_info.status_info_print( f" process_start 0:  {str(e)} ")  
          
def get_folder_size(folder_path):
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(f"""{folder_path}"""):
        for filename in filenames:
            filepath = os.path.join(f"""{dirpath}""", f"""{filename}""")
            try:
                total_size += os.path.getsize(f"""{filepath}""")
            except Exception as e:
                log_info.status_info_print( f"get size : {folder_path} :  {str(e)} ")
    return total_size
            
def search_file_main(Version, path_info, es, jsondata, filterinfolist, sortingenable, tika_server_mode, tika_server_port, counters):
    
    log_info.status_info_print( f"start search_file_main  ")  
    try:       
        gstart_time = time.time()
        
        root_path = jsondata['root_path']
        max_limited = jsondata['max_decompress_limited']
        main_fold = jsondata['main_fold']
        sub_main_fold = jsondata['sub_main_fold'] 
        
        ocrflag = jsondata['ocrflag'] 
                
        datainfopath = jsondata['datainfopath'] 
        source_path = datainfopath['source_path'] 
        target_path = datainfopath['target_path'] 
        issue_path = datainfopath['issue_path']
        
        elasticsearch = jsondata['elasticsearch']   
        elastic_server_index = elasticsearch['server_index']            
        total_list = []
    
        start_time = time.time()
        
        source_path = os.path.join(root_path, source_path) 
    except Exception as e:
        log_info.status_info_print( f"search_file_main :  {str(e)} ")   
            
    try:   
        get_file_count(source_path, filterinfolist, counters)
    except Exception as e:
        log_info.status_info_print( f"get_file_count :  {str(e)} ")   
        
    for file_info in counters.compress_file_list:  
        try:
            info_message = f"0 counters.compress_file_list -  {file_info} "
            # log_info.status_info_print(info_message) 
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.compress_file_list -  {str(e)} "
            log_info.status_info_print(info_message)   
    counters.init_compress_file_list()
                    
    counters.org_count = counters.files_count
    counters.files_count = 0
    
    end_time = time.time()
  
    log_info.status_info_print( f"\n1.{sub_main_fold}  Source file scan Time: %.2f seconds" % (end_time - start_time))

    log_info.status_info_print( f"\nNumber of Source Path files: {counters.org_count}")    

    
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : Number of Source Path files: {counters.org_count}", log_level=logging.INFO)      
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 1. Source file scan Time : {(end_time - start_time):.2f} seconds", log_level=logging.INFO)    
    
    # 압축해제
    start_time = time.time()
    try:   
        decompress_file.decompress_main(source_path, target_path, issue_path, counters, max_limited, jsondata['cpu_use_persent'])
    except Exception as e:
        log_info.status_info_print( f"decompress_main :  {str(e)} ")       

    first_target_decompressed_size = get_folder_size(target_path)    
    end_time = time.time()

    log_info.status_info_print( f"\n2.{sub_main_fold}  decompress Time: %.2f seconds" % (end_time - start_time))
    log_info.status_info_print( f"decompress_count : {counters.decompress_count}")
    
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : decompress_count : {counters.decompress_count}", log_level=logging.INFO)      
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 2. decompress Time : {(end_time - start_time):.2f} seconds", log_level=logging.INFO)    
    #log_info.status_info_print( f"decompress_exception_count : {decompress_file.decompress_exception_count}")

    
    # 파일검색    
    log_info.status_info_print( f"start get file name ")
    start_time = time.time()
    get_file_names(target_path, total_list, filterinfolist, counters, ocrflag)
    end_time = time.time() 
    log_info.status_info_print( f"counters.compress_file_list len : {len(counters.compress_file_list)}")
    for file_info in counters.compress_file_list:  
        try:
            info_message = f"1 counters.compress_file_list -  {file_info} "
            log_info.status_info_print(info_message) 
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.compress_file_list -  {str(e)} "
            log_info.status_info_print(info_message)  
             
    counters.init_compress_file_list()
                    
    log_info.status_info_print( f"\n3.{sub_main_fold}  file count : {counters.files_count}:{len(total_list)},  scan Time: %.2f seconds" % (end_time - start_time))
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 3.  file count : {counters.files_count}, file scan Time : {(end_time - start_time):.2f} seconds", log_level=logging.INFO)  
    
    # list sorting    
    start_time = time.time()
    total_list=list_sorting(total_list, sortingenable, counters)
    end_time = time.time()
    log_info.status_info_print( f"\n3-1.{sub_main_fold}  soring Time: %.2f seconds" % (end_time - start_time))
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 3-1. file sorting scan Time : {(end_time - start_time):.2f} seconds", log_level=logging.INFO)  
     
    file_count = 0    
    counters.target_count  = file_count = len(total_list) # 1차 target count
    log_info.status_info_print( f"\n numbers of after decompression : {file_count}")  
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : numbers of after decompression: {file_count}", log_level=logging.INFO)     
     

    # 파일분석
    start_time = time.time()
    try:   
        # asyncio.run(process_start(total_list, path_info, es, jsondata, "normal", counters))

        process_start(total_list, path_info, es, jsondata, tika_server_mode, "normal", filterinfolist, counters, target_path, issue_path)
    except Exception as e:
        log_info.status_info_print( f"process_start 2:  {str(e)} ")       
    
    gsecond_time = end_time = time.time()

    log_info.status_info_print( f"\n4. {sub_main_fold} file analysis & indexing Time: %.2f seconds" % (end_time - start_time))

    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 4. file analysis & indexing Time: {(end_time - start_time):.2f} seconds", log_level=logging.INFO)  


    mail_root_path = path_info["mail_root_path"]
    try:   
        get_file_count(mail_root_path, filterinfolist, counters)
    except Exception as e:
        log_info.status_info_print( f"get_file_count :  {str(e)} ")      
    counters.files_count = 0        
        
    for file_info in counters.compress_file_list:  
        try:
            info_message = f"2 counters.compress_file_list -  {file_info} "
            # log_info.status_info_print(info_message) 
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.compress_file_list -  {str(e)} "
            log_info.status_info_print(info_message)   
    counters.init_compress_file_list()
        
    # 메일의 첨부파일 분석 : 압축해제 => 파일검색 => 파일분석
    # pst, eml, msg 첨부파일의 압축해제
    
    # mail에 대한 첨부파일 searching 시 압축해제 count 초기화
    decompress_file.decompress_count = 0
    
    start_time = time.time()

    pst_attachment_path = os.path.join(path_info["pst_attachment_path"], "attachment")

    thread_info_list = [
        (pst_attachment_path, pst_attachment_path),
        (path_info["eml_attachment_path"], path_info["eml_attachment_path"]),
        (path_info["msg_attachment_path"], path_info["msg_attachment_path"]),
        (path_info["mbox_attachment_path"], path_info["mbox_attachment_path"]),        
        (path_info["edb_attachment_path"], path_info["edb_attachment_path"]),  
        (path_info["nsf_attachment_path"], path_info["nsf_attachment_path"])  
    ]
    try:   
        with concurrent.futures.ThreadPoolExecutor(max_workers=jsondata["max_thread_num"]) as executor:
            # 각 파일을 독립적으로 처리하는 작업을 제출
            futures = [executor.submit(decompress_file.decompress_main, thread_info[0], thread_info[1], issue_path,  counters, max_limited, jsondata['cpu_use_persent']) for thread_info in thread_info_list]

            # 모든 작업이 완료될 때까지 대기
            concurrent.futures.wait(futures)
            executor.shutdown()
    except Exception as e:
        log_info.status_info_print( f"process_start 3:  {str(e)} ")    

        # 이후 작업을 수행할 코드를 추가하십시오.
    secode_target_decompressed_size = get_folder_size(target_path)  
            
    end_time = time.time()

    log_info.status_info_print( f"\nmail decompress_count : {decompress_file.decompress_count}")
    log_info.status_info_print( f"\n5. {sub_main_fold} decompress of mail(pst,eml,msg) Time: %.2f seconds" % (end_time - start_time))


    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : mail decompress_count : {decompress_file.decompress_count}", log_level=logging.INFO)   
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 5. decompress of mail(pst,eml,msg) Time: {(end_time - start_time):.2f} seconds", log_level=logging.INFO)  
    #log_info.status_info_print( f"mail decompress_exception_count : {decompress_file.decompress_exception_count}")
    
    # 메일 첨부파일 검색
    mail_attachment_list = []  
    start_time = time.time()

    get_file_names(pst_attachment_path, mail_attachment_list, filterinfolist, counters, ocrflag)
    get_file_names(path_info["eml_attachment_path"], mail_attachment_list, filterinfolist, counters, ocrflag)
    get_file_names(path_info["msg_attachment_path"], mail_attachment_list, filterinfolist, counters, ocrflag)
    get_file_names(path_info["mbox_attachment_path"], mail_attachment_list, filterinfolist, counters, ocrflag) 
    get_file_names(path_info["edb_attachment_path"], mail_attachment_list, filterinfolist, counters, ocrflag) 
    get_file_names(path_info["nsf_attachment_path"], mail_attachment_list, filterinfolist, counters, ocrflag)         
    end_time = time.time()   

    file_count = len(mail_attachment_list)
    counters.target_count  += file_count
    
    log_info.status_info_print( f"\nNumber of new total(mail(pst,eml,msg) attachments) files: {file_count}")
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : Number of new total(mail(pst,eml,msg) attachments) files: {file_count}", log_level=logging.INFO)  
                                    
    log_info.status_info_print( f"\n6. {sub_main_fold} search for mail(pst,eml,msg) attachments Time: %.2f seconds" % (end_time - start_time))   
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 6. search for mail(pst,eml,msg) attachments Time: {(end_time - start_time):.2f} seconds", log_level=logging.INFO)  
 
     # list sorting    
    start_time = time.time()
    mail_attachment_list=list_sorting(mail_attachment_list, sortingenable, counters)
    end_time = time.time()
    log_info.status_info_print( f"\n6-1.{sub_main_fold}  soring Time: %.2f seconds" % (end_time - start_time))
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 6-1. file sorting scan Time : {(end_time - start_time):.2f} seconds", log_level=logging.INFO)  
          
    
    # 메일 파일분석, count 초기화
    
    start_time = time.time()
    counters.processor_count = 0
    start_time = time.time()
    try: 
        # asyncio.run(process_start(mail_attachment_list, path_info, es, jsondata, "mail", counters))  
        process_start(mail_attachment_list, path_info, es, jsondata, tika_server_mode, "mail", filterinfolist, counters, target_path, issue_path) 
    except Exception as e:
        log_info.status_info_print( f"process_start 4:  {str(e)} ")       
       
    end_time = time.time() 

    log_info.status_info_print( f"\n7. {sub_main_fold} analysis of mail attachments & indexing Time: %.2f seconds" % (end_time - start_time))
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 7. analysis of mail attachments & indexing Time: {(end_time - start_time):.2f} seconds" , log_level=logging.INFO)  
    
    # 작업 완료 후 최종 target 파일검색    
    target_list = []
    start_time = time.time()
    
    counters.files_count = 0
    
    get_file_count(target_path, filterinfolist, counters)
 
    end_time = time.time()

    log_info.status_info_print( f"\nNumber of Total files after target fold: {counters.files_count}")  
     
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : Number of Total files after target fold: {counters.files_count}", log_level=logging.INFO)  
    
    log_info.status_info_print( f"\n8. {sub_main_fold} Total file scan Time: %.2f seconds" % (end_time - start_time))
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 8. Total file scan Time: {(end_time - start_time):.2f} seconds ", log_level=logging.INFO)
    
    now = datetime.datetime.now()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_info.status_info_print( f"\n9. {sub_main_fold} End time: {current_time} ")
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 9. End time:  {current_time}", log_level=logging.INFO)
        
    gend_time = time.time()
    
    all_time_minutes = (gend_time - gstart_time) / 60
    without_copy_minutes = ((gend_time - gstart_time) - gcopy_time) / 60

    first_step_minutes = ((gsecond_time - gstart_time) - gcopy_time) / 60
    log_info.status_info_print( f"\n10. {sub_main_fold} Total Working Time: {all_time_minutes:.2f} minutes")
    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : 10. Total Working Time: {all_time_minutes:.2f} minutes ", log_level=logging.INFO)
 
    # counters.total_count = counters.fileinfo_analyze_count  + counters.files_exception_count + counters.files_without_extensions  + counters.files_symbolic_count
    analyzed_count = counters.fileinfo_analyze_count - counters.files_exception_count
    
    last_target_path_size = get_folder_size(target_path)   
    
    el_file_path = jsondata["elasticsearch"]['el_file_path']
    el_mail_path = jsondata["elasticsearch"]['el_mail_path']    
    el_info_path = jsondata["elasticsearch"]['el_info_path'] 
    
    el_file_target_path = jsondata["elasticsearch"]['el_file_target_path']   
    
    el_file_target_path = os.path.join(root_path, el_file_target_path)
    
    es_file_size = 0 
    es_mail_size = 0   
    
    try:   
                        
        if os.path.exists(el_file_target_path):
        
            dir_path = os.path.join(el_file_target_path, el_file_path, sub_main_fold)
            es_file_size = get_folder_size(dir_path)  
            
            dir_path = os.path.join(el_file_target_path, el_mail_path, sub_main_fold)
            es_mail_size = get_folder_size(dir_path)   
               
    except Exception as e:

        info_message = f"Exception get size from es: {str(e)}"
        log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)  
                     
                        
    status_info = {
        "source_path_name":sub_main_fold,
        "total_working_time_m": all_time_minutes,
        "without_copy_working_time_m": without_copy_minutes,
        "first_tep_end_time_m": first_step_minutes,
        "files": {
            "source_path_file_count": counters.org_count, #source path file count
            "last_target_path_total_count":counters.files_count, #최종 target fold의 파일 카운터
            
            "target_path_file_count": counters.target_count,# try analyzer count
            
            "file_analyze_count": counters.fileinfo_analyze_count, 
            "files_without_extensions": counters.files_without_extensions,  # there is no extension
            "files_symbolic_count": counters.files_symbolic_count,
            "file_ext_count": counters.file_ext_list_count, 
            "file_issue_list_count": counters.file_issue_list_count,
             
            "analyzed_count":analyzed_count,
            "analyzed_exception_count": counters.files_exception_count,
            "analyzer_issue_list_count": counters.analyzer_issue_list_count,
                                     
            "decompress_count" : counters.decompress_count,
            "decompress_exception_count" : counters.decompress_exception_count,
            
            "files_index_count": counters.files_index_count - 1,
            "files_index_exception_count": counters.files_index_exception_count
           
        },
        "mails": {
            
            "pst files": counters.pst_count,
            "pst_exception_files": counters.pst_exception_count,
            
            "msg_in_the_pst_count": counters.pst_msg_count,
            "msg_in_the_pst_exception_files": counters.pst_msg_exception_count,
            
            "eml_files": counters.eml_count,
            "eml_exception_files": counters.eml_exception_count,
            
            "msg_files": counters.msg_count,
            "msg_exception_files": counters.msg_exception_count,
            
            "mbox_files": counters.mbox_count,
            "mbox_exception_files": counters.mbox_exception_count,
            
            "edb_files": counters.edb_count,
            "edb_exception_files": counters.edb_exception_count,
            
            "nsf_files": counters.nsf_count,
            "nsf_exception_files": counters.nsf_exception_count,
                                    
            "attached_file_count": counters.attached_file_count,
            
            "files_index_exception_count": counters.files_index_exception_count,
            
            "pst_index_exception_count": counters.pst_index_exception_count,
            "eml_index_exception_count": counters.eml_index_exception_count,
            "msg_index_exception_count": counters.msg_index_exception_count,
            "mbox_index_exception_count": counters.mbox_index_exception_count,
            
            "total_mail_count": counters.pst_msg_count + counters.eml_count + counters.msg_count + counters.mbox_count + counters.edb_count + counters.nsf_count,
            "total_excetpion_mail_count": counters.pst_msg_exception_count + counters.eml_exception_count + counters.msg_exception_count + counters.mbox_exception_count  + counters.edb_exception_count + counters.nsf_exception_count           
        },
        "size":{
            "source_path_size": path_info['source_path_size'],
            "target_path_size": path_info['target_path_size'],
            "frist_target_decompressed_size": first_target_decompressed_size,
            "second_target_decompressed_size": secode_target_decompressed_size,
            "last_target_path_size": last_target_path_size,
            "es_file_size": es_file_size,
            "es_mail_size": es_mail_size,  
            
        },
        "summary":f"last_count({counters.files_count}) = target_count({counters.target_count}) + file_ext({counters.file_ext_list_count}) + file_symbolic({counters.files_symbolic_count})+ file_issue({counters.file_issue_list_count})+ pst_eml_count({counters.pst_msg_count})"
    }

    info_status_info = {
        "status_info": status_info
    }   

    log_info.status_info_print(f"""{Version} : {status_info}""") 
    
    if es is not None: # indexing document
         
        el_index_status_info_name = jsondata["elasticsearch"]["el_index_status_info_name"] + "_" +  sub_main_fold
        log_info.debug_print( f"\n {sub_main_fold} Indexing status info : {el_index_status_info_name} - {info_status_info}\n")
        es_status = es_mng.es_index_data(es, jsondata, el_index_status_info_name, info_status_info, "indexing status info", counters)
        
        if es_status == False:
            es = es_mng.es_index_connection(jsondata)
            
            if es is None:
                # sys.exit(1)
                pass
            else: 
                es_status = es_mng.es_index_data(es, jsondata, el_index_status_info_name, info_status_info, "indexing status info", counters)
                if es_status == False:
                    # sys.exit(1) 
                    pass
                    
    # issue_listfile_info = {
    #     "file_issue_list_info":counters.file_issue_list
    # }                
    # if es is not None: # indexing issue info
         
    #     el_index_file_issue_list_info = jsondata["elasticsearch"]["el_index_file_issue_list_info"] + "_" + sub_main_fold
    #     log_info.debug_print( f"\n {sub_main_fold} Indexing issue fileList info : {el_index_file_issue_list_info} - {counters.file_issue_list}\n")
        
    #     es_status = es_mng.es_index_data(es, jsondata, el_index_file_issue_list_info, issue_listfile_info, "Indexing issue fileList info", counters)
        
    #     if es_status == False:
    #         es = es_mng.es_index_connection(jsondata)
            
    #         if es is None:
    #             # sys.exit(1) 
    #             pass
    #         else: 
    #             es_status = es_mng.es_index_data(es, jsondata, el_index_file_issue_list_info, issue_listfile_info, "Indexing issue fileList info", counters)
    #             if es_status == False:
    #                 # sys.exit(1) 
    #                 pass  
    for file_info in counters.file_issue_list:  
        try:
            # info_message = f"counters.file_issue_list -  {file_info} "
            # log_info.status_info_print(info_message) 
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.file_issue_list -  {str(e)} "
            log_info.status_info_print(info_message)      
                
    # analyzer_issue_listfile_info = {
    #     "analyzer_issue_list_info":counters.analyzer_issue_list
    # }                
    # if es is not None: # indexing issue info
         
    #     el_index_file_issue_list_info = jsondata["elasticsearch"]["el_index_file_issue_list_info"] + "_" + sub_main_fold
    #     log_info.debug_print( f"\n {sub_main_fold} Indexing issue fileList info : {el_index_file_issue_list_info} - {counters.analyzer_issue_list}\n")
        
    #     es_status = es_mng.es_index_data(es, jsondata, el_index_file_issue_list_info, analyzer_issue_listfile_info, "Indexing issue fileList info", counters)
        
    #     if es_status == False:
    #         es = es_mng.es_index_connection(jsondata)
            
    #         if es is None:
    #             # sys.exit(1) 
    #             pass
    #         else: 
    #             es_status = es_mng.es_index_data(es, jsondata, el_index_file_issue_list_info, analyzer_issue_listfile_info, "Indexing issue fileList info", counters)
    #             if es_status == False:
    #                 # sys.exit(1) 
    #                 pass    
                         
    for file_info in counters.analyzer_issue_list:  
             
        try:
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.analyzer_issue_list -  {str(e)} "
            log_info.status_info_print(info_message)                         
                         
    # counters.file_ext_list = list(set(counters.file_ext_list))
    # filter_ext_list_info = {
    #     "filter_ext_list_info": counters.file_ext_list
    # }                         
    # if es is not None: # indexing filter ext list info
         
    #     el_index_filter_ext_list_info = jsondata["elasticsearch"]["el_index_filter_ext_list_info"] + "_" + sub_main_fold
    #     log_info.debug_print( f"\n {sub_main_fold} Indexing fileter ext list info : {el_index_filter_ext_list_info} - {counters.file_ext_list}\n")
        
    #     es_status = es_mng.es_index_data(es, jsondata, el_index_filter_ext_list_info, filter_ext_list_info, "Indexing fileter ext list info", counters)
        
    #     if es_status == False:
    #         es = es_mng.es_index_connection(jsondata)
            
    #         if es is None:
    #             # sys.exit(1) 
    #             pass
    #         else: 
    #             es_status = es_mng.es_index_data(es, jsondata, el_index_filter_ext_list_info, filter_ext_list_info, "Indexing fileter ext list info", counters)
    #             if es_status == False:
    #                 # sys.exit(1) 
    #                 pass  
                
    for file_info in counters.file_ext_list:  
        
        try:
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.file_ext_list -  {str(e)} "
            log_info.status_info_print(info_message)  
                              
                    
    status_log_info = {
        "status_log_info":f"{log_info.counter_log.log_info}"
    }                                                    
    if es is not None: # indexing log info
         
        el_index_status_log_info = jsondata["elasticsearch"]["el_index_status_log_info"] + "_" + sub_main_fold
        log_info.debug_print( f"\n {sub_main_fold} Indexing status log info : {el_index_status_log_info} - {log_info.counter_log.log_info}\n")
        
        es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_log_info, "Indexing status log info", counters)
        
        if es_status == False:
            es = es_mng.es_index_connection(jsondata)
            
            if es is None:
                # sys.exit(1) 
                pass
            else: 
                es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_log_info, "Indexing status log info", counters)
                if es_status == False:
                    # sys.exit(1) 
                    pass  
                 

    # status_com_info = {
    #     "com": counters.compress_file_list
    # }                                                    
    # if es is not None: # indexing log info
         
    #     el_index_status_log_info = jsondata["elasticsearch"]["el_index_file_issue_list_info"] + "_" + sub_main_fold
    #     log_info.debug_print( f"\n {sub_main_fold} compressed files list info: {el_index_status_log_info} - {status_com_info}\n")
        
    #     es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_com_info, "Indexing status log info", counters)
        
    #     if es_status == False:
    #         es = es_mng.es_index_connection(jsondata)
            
    #         if es is None:
    #             # sys.exit(1) 
    #             pass
    #         else: 
    #             es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_com_info, "Indexing status log info", counters)
    #             if es_status == False:
    #                 # sys.exit(1) 
    #                 pass   

    for file_info in counters.compress_file_list:  
        try:
            info_message = f"3 counters.compress_file_list -  {file_info} "
            # log_info.status_info_print(info_message) 
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.compress_file_list -  {str(e)} "
            log_info.status_info_print(info_message)       
        
    # status_crp_info = {
    #     "cry": counters.cry_file_list
    # }                                                    
    # if es is not None: # indexing log info
         
    #     el_index_status_log_info = jsondata["elasticsearch"]["el_index_file_issue_list_info"] + "_" + sub_main_fold
    #     log_info.debug_print( f"\n {sub_main_fold} crypto files llist info : {el_index_status_log_info} - {status_crp_info}\n")
        
    #     es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_crp_info, "Indexing status log info", counters)
        
    #     if es_status == False:
    #         es = es_mng.es_index_connection(jsondata)
            
    #         if es is None:
    #             # sys.exit(1) 
    #             pass
    #         else: 
    #             es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_crp_info, "Indexing status log info", counters)
    #             if es_status == False:
    #                 # sys.exit(1) 
    #                 pass   

    # status_drm_info = {
    #     "drm":counters.drm_file_list
    # }                                                    
    # if es is not None: # indexing log info
         
    #     el_index_status_log_info = jsondata["elasticsearch"]["el_index_file_issue_list_info"] + "_" + sub_main_fold
    #     log_info.debug_print( f"\n {sub_main_fold} Indexing status log info : {el_index_status_log_info} - {status_drm_info}\n")
        
    #     es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_drm_info, "Indexing status log info", counters)
        
    #     if es_status == False:
    #         es = es_mng.es_index_connection(jsondata)
            
    #         if es is None:
    #             # sys.exit(1) 
    #             pass
    #         else: 
    #             es_status = es_mng.es_index_data(es, jsondata, el_index_status_log_info, status_drm_info, "Indexing status log info", counters)
    #             if es_status == False:
    #                 # sys.exit(1) 
    #                 pass  
                 
    for file_info in counters.drm_file_list:  
        try:
            main_ut.es_indexing_ext(es, jsondata,  main_fold, file_info['file_path'], file_info['error_message'], es_mng, counters)     
        except Exception as e:
            info_message = f"counters.drm_file_list -  {str(e)} "
            log_info.status_info_print(info_message)     
        
    if es:
        if elastic_server_index == 1 or elastic_server_index == 2: 
            es.close()
        elif elastic_server_index == 3: 
            es = None           
    # file_es.end_save_data_to_file(jsondata["elasticsearch"], counters)
    # time.sleep(5)
         
    return info_status_info


def copy_directory(source_path, target_path, log_info, fold_name):
    try:
        new_target_path = ""
        new_source_path = os.path.join(source_path, fold_name)

        if os.path.isdir(new_source_path):
            fold_name = main_ut.replace_brackets_onely_file_name(fold_name)
            new_target_path = os.path.join(target_path, fold_name)
            
            # log_info.status_info_print(f"copy starting.... 1: from {new_source_path} to {new_target_path}")

            copy_start_time = time.time()
            shutil.copytree(new_source_path, new_target_path)
            copy_end_time = time.time()

            log_info.status_info_print(f"\n  {new_source_path}  copy times : {(copy_end_time - copy_start_time)} sec")
        else:
            shutil.copy(new_source_path, target_path)
            # log_info.status_info_print(f"copy file....  2: from {new_source_path} to {target_path}")

    except shutil.Error as e:
        for src, dst, msg in e.args[0]:
            if "No such file or directory" in msg:
                log_info.status_info_print(f"there is no Error in msg: {str(e)}")
            else:
                log_info.status_info_print(f"error : {str(e)}")
    except Exception as e:            
        log_info.status_info_print(f"copy_directory {source_path} error : {str(e)}")  
        
               
#if __name__ == "__main__":
def sub_main_processor(Version, filterinfolist):
    try:
        strVersion = f"start sub_main_processor : Ver-{Version}"
        log_info.status_info_print( strVersion)   
            
        now = datetime.datetime.now()

        
        # PST 파일 경로
        json_data  =  main_ut.config_reading('config.json')
        
        if json_data != None:
            
            
            # 2023.08.11
            # json path 변경
            # root_path 가져오기
            root_path = json_data['root_path']

            if not os.path.exists(root_path):
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_info.status_info_print( f"\n Exit there is no root path  time: ", {current_time})
                sys.exit(1)
                
            # addflag = json_data['addflag']   
                    
            pst2eml = json_data['pst2eml']
            edbflag = json_data['edbflag']
            nsfflag = json_data['nsfflag'] 
            ocrflag = json_data['ocrflag']        
                
            # sub_path 가져오기
            datainfopath = json_data['datainfopath']
            mail = json_data['mail']
            elasticsearch = json_data['elasticsearch']
            elastic_server_index = elasticsearch['server_index']
            max_limit_siz = elasticsearch['el_file_max_size']
            counters = CountersWithLock(max_limit_siz)  
                
            # tika path info
            tika_app = json_data['tika_app']
  
            tika_server_mode = tika_app['tika_server_mode']            
            tika_app_jar_path = tika_app['tika_jar_path']
            tika_app_jar_name = tika_app['tika_jar_name']
            tika_server_ip = tika_app['tika_server_ip']
            tika_server_port = tika_app['tika_server_port']
            max_tika_thread_num = tika_app['max_tika_thread_num']
            tika_xms = 10
            tika_xmx = 5120    
                
            try:
                tika_xms = tika_app['tika_xms']
                tika_xmx = tika_app['tika_xmx']
            except OSError as e:
                log_info.status_info_print( f"used default tika memory: {tika_xms} - {tika_xmx}")    
            except Exception as e:            
                log_info.status_info_print(f"Exception  error : {str(e)} - {tika_xms} - {tika_xmx}")
                                                 
            max_decompress_limited = json_data['max_decompress_limited']   

                
            max_thread_num = json_data['max_thread_num']   
            # if (current_thread_count//5) >  max_thread_num:
            #     max_thread_num  = (current_thread_count//5)
        

            # Tika JAR 파일의 경로
            # tika_jar_path = "./tika-app-2.9.0.jar"  # tika-app.jar 파일의 실제 경로로 변경해야 합니다
            tika.initVM()
            
            # tika_server_endpoint = f"http://127.0.0.1:{tika_server_port}/tika"  # 사용하려는 서버의 엔드포인트
            tika_server_endpoint = f"""http://{tika_server_ip}:{tika_server_port}/tika"""  # 사용하려는 서버의 엔드포인트
                
            current_directory = os.getcwd()
            # print("현재 디렉토리:", current_directory)
            if tika_app_jar_path is None or not tika_app_jar_path.strip():
                tika_app_jar_path  = os.path.join(current_directory, tika_app_jar_name)
            else:
                tika_app_jar_path = tika_app_jar_path
                
            # Tika JAR 파일을 환경 변수로 설정
            os.environ['TIKA_PATH'] = tika_app_jar_path
            os.environ['TIKA_LOG_PATH'] = '/data'
                
            main_fold =  json_data["main_fold"] # this is for root path of main
            sub_main_fold = json_data["sub_main_fold"]
                    
            source_path = datainfopath['source_path']
            target_path = datainfopath['target_path']
            issue_path = datainfopath['issue_path']
            
            source_path = os.path.join(root_path, source_path)
            target_path = os.path.join(root_path, target_path)
            issue_path = os.path.join(root_path, issue_path)        
            
            
            source_path_size = get_folder_size(source_path)
            
            CPU_THRESHOLD = json_data["cpu_use_persent"]
            sortingenable = json_data["sorting_mode"]  
            
            # 공통
            mail_root_path = mail['mail_root_path']
            mail_root_path = os.path.join(target_path, mail_root_path)
            
            #start Tika Server
            if tika_server_mode:  
                # port_number = 9998  # 사용하려는 포트 번호
                kill_tika_by_port(sub_main_fold, tika_server_port)
                time.sleep(1)
                # tika_xms = 512
                # tika_xmx = 5120
                java_command = f"nohup java -Xms{tika_xms}m -Xmx{tika_xmx}m -jar"
                start_tika_server(tika_server_port, java_command)   
                                
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_info.status_info_print( f"\n 0. {sub_main_fold} : start time: {current_time} ")
                                    
            if source_path == target_path:
            # 복사 없이 작업 진행 / 동일폴드에 mail attacment file 삭제 
            # rename for same fold without special char
                main_ut.rename_folders(source_path) 
    
            target_path_size = get_folder_size(target_path)   
                        
            # 공통
            try:
                if os.path.exists(mail_root_path):
                    shutil.rmtree(mail_root_path)
                    log_info.status_info_print( f"mail path deleted successed : {mail_root_path}" )
            except OSError as e:
                log_info.status_info_print( f"failed mail path: {str(e)} - {mail_root_path}")
            except Exception as e:            
                log_info.status_info_print(f"Exception  error : {str(e)} - {mail_root_path}")                             
            try:       
                if os.path.exists(issue_path):    
                    shutil.rmtree(issue_path)
                    log_info.status_info_print( f"delete issue path successed : {issue_path}")
                
                # issue path create
                os.makedirs(issue_path)
                
            except OSError as e:
                log_info.status_info_print( f" there is no path info OSError : {str(e)} ", issue_path)  

            except Exception as e:            
                log_info.status_info_print(f"Exception  error : {str(e)} - {issue_path}")  
                                       
            all_path_list = []
            target_fold_list = []
            
            if os.path.exists(source_path): 
                all_path_list = os.listdir(source_path)    
                                            
            file_masking_path = ""

            masking_path, last_folder = os.path.split(target_path)
            
            file_masking_path = masking_path
            
            pst_attachment_path = os.path.join(mail_root_path, mail['pst_attachment_path'])
            eml_attachment_path = os.path.join(mail_root_path, mail['eml_attachment_path'])
            msg_attachment_path = os.path.join(mail_root_path, mail['msg_attachment_path']) 
            mbox_attachment_path = os.path.join(mail_root_path, mail['mbox_attachment_path']) 
            edb_attachment_path = os.path.join(mail_root_path, mail['edb_attachment_path'])    
            nsf_attachment_path = os.path.join(mail_root_path, mail['nsf_attachment_path'])           

            if not os.path.exists(pst_attachment_path):
                new_path = os.path.join(pst_attachment_path, "attachment", sub_main_fold)
                os.makedirs(new_path)

            if not os.path.exists(eml_attachment_path):
                new_path = os.path.join(eml_attachment_path, "attachment", sub_main_fold)
                os.makedirs(new_path)

            if not os.path.exists(msg_attachment_path):
                new_path = os.path.join(msg_attachment_path, "attachment", sub_main_fold)
                os.makedirs(new_path) 
                    
            if not os.path.exists(mbox_attachment_path):
                new_path = os.path.join(mbox_attachment_path, "attachment", sub_main_fold)
                os.makedirs(new_path)   
                    
            if not os.path.exists(edb_attachment_path):
                new_path = os.path.join(edb_attachment_path, "attachment", sub_main_fold)
                os.makedirs(new_path)  
                    
            if not os.path.exists(nsf_attachment_path):
                new_path = os.path.join(nsf_attachment_path, "attachment", sub_main_fold)
                os.makedirs(new_path)                                                  

            
            el_file_path = elasticsearch['el_file_path']
            el_mail_path = elasticsearch['el_mail_path']    
            el_info_path = elasticsearch['el_info_path'] 
            
            el_file_target_path = elasticsearch['el_file_target_path']   
            
            dir_path = None
            
            try:   
                            
                if not os.path.exists(el_file_target_path):
                    logmsg = "make target_path + "
                    os.makedirs(el_file_target_path)
                
                    dir_path = os.path.join(el_file_target_path, el_file_path, sub_main_fold)
                    log_info.status_info_print(f" dir_path : {dir_path}")
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                        
                    dir_path = os.path.join(el_file_target_path, el_mail_path, sub_main_fold)
                    log_info.status_info_print(f" dir_path : {dir_path}")
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                        
                    dir_path = os.path.join(el_file_target_path, el_info_path, sub_main_fold)
                    log_info.status_info_print(f" dir_path : {dir_path}")
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                                                        
            except OSError as e:
                log_info.status_info_print( f" el path OSError : {str(e)} - {dir_path}")    
            except Exception as e:            
                log_info.status_info_print(f"Exception  error : {str(e)} - {dir_path}")                     
            # connection es 
            es = es_mng.es_index_connection(json_data)
            
            if es is not None: # indexing config info
                
                json_config_info = {"config": ""}
                json_config_info["config"] = json_data
                es_status = es_mng.es_index_data(es, json_data, 'el_index_config_name', json_config_info, "config info", counters)
                # es_status = es_mng.es_index_data(es, json_data, 'el_index_config_name', f"""{json_config_info}""", "config info", counters)
                if es_status == False:
                    log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : exit process, because of indexing config info", log_level=logging.INFO)
                    # sys.exit(1) 
                    pass       
            else:
                log_info.log_with_function_name(log_info.info_logger, f"\n {sub_main_fold} : es connection is faild", log_level=logging.ERROR)
                # sys.exit(1) 
                pass             
                                    
            # 생성된 경로 출력

            # log_info.status_info_print( f"file_masking_path : {file_masking_path}")
            
            # log_info.status_info_print( f"root_path : {root_path}")

            # log_info.status_info_print( f"source_path : {source_path}")
            # log_info.status_info_print( f"target_path : {target_path}")
            # log_info.status_info_print( f"issue_path : {issue_path}")
                

            # if tika_server_mode: 
            #     log_info.status_info_print( f"tika_server_port : {tika_server_port}")
            # else:
            #     log_info.status_info_print( f"tika_app_jar_path : {tika_app_jar_path}")
            #     log_info.status_info_print( f"tika_app_jar_name : {tika_app_jar_name}")
            #     log_info.status_info_print( f"max_tika_thread_num: {max_tika_thread_num}")
            
            # log_info.status_info_print( f"max_thread_num : {max_thread_num}")
            # log_info.status_info_print( f"max_decompress_limited : {max_decompress_limited}")        

            # log_info.status_info_print( f"pst_attachment_path : {pst_attachment_path}")
            # log_info.status_info_print( f"eml_attachment_path : {eml_attachment_path}")
            # log_info.status_info_print( f"msg_attachment_path : {msg_attachment_path}")
            # log_info.status_info_print( f"mbox_attachment_path : {mbox_attachment_path}")
            # log_info.status_info_print( f"edb_attachment_path : {edb_attachment_path}")
            # log_info.status_info_print( f"nsf_attachment_path : {nsf_attachment_path}")
                    
            json_data["tika_app"]["tika_server_endpoint"]=  tika_server_endpoint 
            json_data["tika_app"]["tika_app_jar_path"]=  tika_app_jar_path 
            json_data["tika_app"]["tika_app_jar_name"]=  tika_app_jar_name         
            path_info = {
                
                "file_masking_path":file_masking_path,
                
                "mail_root_path":mail_root_path,
                "pst_attachment_path": pst_attachment_path,
                "eml_attachment_path": eml_attachment_path, 
                "msg_attachment_path": msg_attachment_path,
                "mbox_attachment_path": mbox_attachment_path,
                "edb_attachment_path": edb_attachment_path,
                "nsf_attachment_path": nsf_attachment_path,
                                
                "source_path_size": source_path_size,
                "target_path_size": target_path_size,
                "max_decompress_limited": max_decompress_limited,

            }
            
            info_status_info = None
            try:  
                info_status_info = search_file_main(Version,  path_info, es, json_data, filterinfolist, sortingenable, tika_server_mode, tika_server_port, counters)
            except Exception as e:            
                log_info.status_info_print(f"[sub_main_processor] sub_main_fold : Exception  error : {str(e)} - {path_info}") 
                 
            log_info.status_info_print( f"tika_timeout_count : {counters.tika_timeout_count}")
            
            if tika_server_mode:
                kill_tika_by_port(sub_main_fold, tika_server_port)    
                          
            # if ocrflag:
            #     ocr_mag.main_ocr(Version, json_data)
                    
            return info_status_info
    except Exception as e:            
        log_info.status_info_print(f"sub_main_processor  error : {str(e)}")   
        
  
        
if __name__ == "__main__":
    
    counters = CountersWithLock(0)  
    now = datetime.datetime.now()
    gstart_time = time.time()    
    # path 경로
    json_data =  main_ut.config_reading('config.json')

    if json_data != None:  
            
        Version = json_data["ver"]
        filterinfolist = main_ut.display_filter_config_info(json_data)
        
        sub_main_processor(Version, filterinfolist)