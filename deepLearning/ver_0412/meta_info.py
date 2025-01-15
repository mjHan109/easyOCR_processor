# -*- coding: utf-8 -*-
# pip install email
# elasticsearch oss version 일때 아래의 library로 설치 해야함.
# pip install elasticsearch==7.10.1

from collections import Counter
import os
import re
import time
import json
import tika
import threading
import datetime
import uuid
import inspect
import io

import logging  # logging 모듈을 import
import log_info  # log_info 모듈을 임포트하여 설정을 공유

class CountersWithLock:
    def __init__(self, buffer_size_limit):
        
        self.rename_count = 0
        self.rename_count_lock = threading.Lock()
               
        self.org_count = 0
        self.org_count_lock = threading.Lock()

        self.target_count = 0  # 분석 대상 파일수(압축해제후)
        self.target_count_lock = threading.Lock()
        
        self.files_count = 0
        self.files_count_lock = threading.Lock()
                        
        self.processor_count = 0
        self.processor_count_lock = threading.Lock()

        self.files_exception_count = 0
        self.files_exception_count_lock = threading.Lock()
        
        self.files_symbolic_count = 0
        self.files_symbolic_count_lock = threading.Lock()
        
        self.files_index_count = 0
        self.files_index_count_lock = threading.Lock()
                      
        self.files_without_extensions = 0
        self.files_without_extensions_lock = threading.Lock()
        
        self.files_index_count = 0
        self.files_index_count_lock = threading.Lock()
        
        self.files_index_exception_count = 0
        self.files_index_exception_count_lock = threading.Lock()

        self.fileinfo_analyze_count = 0
        self.fileinfo_analyze_count_lock = threading.Lock()

        self.pst_count = 0
        self.pst_count_lock = threading.Lock()

        self.msg_count = 0
        self.msg_count_lock = threading.Lock()

        self.eml_count = 0
        self.eml_count_lock = threading.Lock()
        
        self.mbox_count = 0
        self.mbox_count_lock = threading.Lock()     
           
        self.hwp_hwpx_count = 0
        self.ehwp_hwpx_count_lock = threading.Lock()       

        self.pst_exception_count = 0
        self.pst_exception_count_lock = threading.Lock()

        self.msg_exception_count = 0
        self.msg_exception_count_lock = threading.Lock()

        self.eml_exception_count = 0
        self.eml_exception_count_lock = threading.Lock()

        self.pst_index_exception_count = 0
        self.pst_index_exception_count_lock = threading.Lock()

        self.msg_index_exception_count = 0
        self.msg_index_exception_count_lock = threading.Lock()

        self.eml_index_exception_count = 0
        self.eml_index_exception_count_lock = threading.Lock()
        
        self.mbox_index_exception_count = 0
        self.mobx_index_exception_count_lock = threading.Lock()
        
        self.pst_msg_count = 0
        self.pst_msg_count_lock = threading.Lock()

        self.pst_msg_exception_count = 0
        self.pst_msg_exception_count_lock = threading.Lock()

        self.pst_msg_index_count = 0
        self.pst_msg_index_count_lock = threading.Lock()
        
        self.pst_msg_index_exception_count = 0
        self.pst_msg_index_exception_count_lock = threading.Lock()

        self.eml_msg_count = 0
        self.eml_msg_count_lock = threading.Lock()

        self.eml_msg_exception_count = 0
        self.eml_msg_exception_count_lock = threading.Lock()

        self.eml_msg_index_count = 0
        self.eml_msg_index_count_lock = threading.Lock()
        
        self.eml_msg_index_exception_count = 0
        self.eml_msg_index_exception_count_lock = threading.Lock()
       
        self.mbox_msg_count = 0
        self.mbox_msg_count_lock = threading.Lock()

        self.mbox_exception_count = 0
        self.mbox_exception_count_lock = threading.Lock()
        
        self.mbox_msg_exception_count = 0
        self.mbox_msg_exception_count_lock = threading.Lock()
        
        self.mbox_msg_index_count = 0
        self.mbox_msg_index_count_lock = threading.Lock()
        
        self.mbox_msg_index_exception_count = 0
        self.mbox_msg_index_exception_count_lock = threading.Lock()
                
        self.attached_file_count = 0        
        self.attached_file_count_lock = threading.Lock()

            
        self.total_count = 0
        self.total_count_lock = threading.Lock()

        self.compress_count = 0
        self.compress_file_list = []
        self.add_compress_file_list_lock = threading.Lock()
        self.compress_count_lock = threading.Lock()
        
        self.decompress_count = 0
        self.decompress_count_lock = threading.Lock()   
             
        self.decompress_exception_count = 0
        self.decompress_exception_count_lock = threading.Lock()
        self.add_decompress_exception_count_lock = threading.Lock()
        
        self.file_issue_list_count = 0
        self.file_issue_list_count_lock = threading.Lock() 
        
        self.file_issue_list = []       
        self.add_file_issue_list_lock = threading.Lock()   
        
        self.analyzer_issue_list_count = 0        
        self.analyzer_issue_list_count_lock = threading.Lock()         
        self.analyzer_issue_list = []
        self.add_analyzer_issue_list_lock = threading.Lock()  
        
        self.log_info = []
        self.add_log_info_lock = threading.Lock()  
        
        self.file_ext_list_count = 0
        self.file_ext_list = []
        self.file_ext_list_lock = threading.Lock()  
        self.add_file_ext_list_lock = threading.Lock()  
                
        self.buffer_size_limit = buffer_size_limit
        
        self.file_buffer = io.StringIO()
        self.file_lock = threading.Lock()
        self.mail_buffer = io.StringIO()
        self.mail_lock = threading.Lock()
        
        self.info_buffer = io.StringIO()
        self.info_lock = threading.Lock()  
                      
        self.add_debugging_info_lock = threading.Lock()
        self.file_path_info = {}
        
        self.tika_count_lock = threading.Lock()
        self.tika_count = 0 
                        
    def file_add_to_buffer(self, data):
        with self.file_lock:
            self.file_buffer.write(json.dumps(data,ensure_ascii=False, indent=4) + '\n')

    def file_save_buffer_to_file(self, file_path):
        with self.file_lock:
            self.file_buffer.seek(0)
            # with open(file_path, 'a') as file:
            #     file.write(self.buffer.read())
            try:     
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(self.file_buffer.read())
            except Exception as e:
                info_message = f"Exception : {self.file_buffer} {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)  
                                
            self.file_buffer = io.StringIO()    
             
    def mail_add_to_buffer(self, data):
        with self.mail_lock:
            self.mail_buffer.write(json.dumps(data, ensure_ascii=False, indent=4) + '\n')
            
    
    def mail_save_buffer_to_file(self, file_path):
        with self.mail_lock:
            self.mail_buffer.seek(0)
            # with open(file_path, 'a') as file:
            #     file.write(self.buffer.read())
            try:     
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(self.mail_buffer.read())
            except Exception as e:
                info_message = f"Exception : {self.mail_buffer} {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)  

            self.mail_buffer = io.StringIO()  
            
    def info_add_to_buffer(self, data):
        with self.info_lock:
            self.info_buffer.write(json.dumps(data, ensure_ascii=False, indent=4) + '\n')

    def info_save_buffer_to_file(self, file_path):
        with self.info_lock:
            self.info_buffer.seek(0)
            # with open(file_path, 'a') as file:
            #     file.write(self.buffer.read())
            try:     
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(self.info_buffer.read())
            except Exception as e:
                info_message = f"Exception : {self.info_buffer} {str(e)}"
                log_info.log_with_function_name(log_info.logger, info_message, log_level=logging.ERROR)                 
            self.info_buffer = io.StringIO()  
                                         
    # 각 변수에 대한 증가 함수들 구현
    def add_analyzer_issue_list(self, info):
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        code = caller_frame.f_code
        function_name = code.co_name
        with self.add_analyzer_issue_list_lock:
            info["function_name"] = function_name
            self.analyzer_issue_list.append(info)
            
           
    # 각 변수에 대한 증가 함수들 구현
    def add_file_issue_list(self, info):
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        code = caller_frame.f_code
        function_name = code.co_name
        with self.add_file_issue_list_lock:
            info["function_name"] = function_name
            self.file_issue_list.append(info)
            
    def add_compress_file_list(self, info):
        with self.add_compress_file_list_lock:
            self.compress_file_list.append(f"""{info}""")   
                                  
    def add_file_ext_list(self, info):
        with self.add_file_ext_list_lock:
            self.file_ext_list.append(f"""{info}""")    
                 
    def add_log_file_list(self, info):
        with self.add_log_info_lock:
            self.log_info.append(f"""{info}""")   

    def add_debugging_file_list(self, key, info):
        with self.add_debugging_info_lock:
            self.file_path_info[key] = f"""{info}"""  
            
    def init_debugging_file_list(self):
        with self.add_debugging_info_lock:
            self.file_path_info = {}   
                  
    def del_debugging_file_list(self, key):
        with self.add_debugging_info_lock:
            self.file_path_info.pop(key)  
                                         
    def increment_rename_count(self):
        with self.rename_count_lock:
            self.rename_count += 1  
            
    def increment_tika_count(self):
        with self.tika_count_lock:
            self.tika_count += 1  
            
    def decrement_tika_count(self):
        with self.tika_count_lock:
            self.tika_count -= 1  
                                             
    def increment_processor_count(self):
        with self.processor_count_lock:
            self.processor_count += 1  
                       
    def increment_files_count(self):
        with self.files_count_lock:
            self.files_count += 1   
                           
    def increment_file_issue_list_count(self):
        with self.file_issue_list_count_lock:
            self.file_issue_list_count += 1 
              
    def increment_analyzer_issue_list_count(self):
        with self.analyzer_issue_list_count_lock:
            self.analyzer_issue_list_count += 1   
                    
    def increment_ext_list_count(self):
        with self.file_ext_list_lock:
            self.file_ext_list_count += 1             
                          
    def increment_org_count(self):
        with self.org_count_lock:
            self.org_count += 1
       
    def increment_target_count(self):
        with self.target_count_lock:
            self.target_count += 1

    def increment_hwp_hwpx_count(self):
        with self.hwp_hwpx_count_lock:
            self.hwp_hwpx_count += 1           

    def increment_files_exception_count(self):
        with self.files_exception_count_lock:
            self.files_exception_count += 1
            
    def increment_files_symbolic_count(self):
        with self.files_symbolic_count_lock:
            self.files_symbolic_count += 1     
            
    def increment_files_without_extensions(self):
        with self.files_without_extensions_lock:
            self.files_without_extensions += 1
            
    def increment_files_index_count(self):
        with self.files_index_count_lock:
            self.files_index_count += 1
            
    def increment_files_index_exception_count(self):
        with self.files_index_exception_count_lock:
            self.files_index_exception_count += 1

    def increment_fileinfo_analyze_count(self):
        with self.fileinfo_analyze_count_lock:
            self.fileinfo_analyze_count += 1

    def increment_pst_count(self):
        with self.pst_count_lock:
            self.pst_count += 1

    def increment_msg_count(self):
        with self.msg_count_lock:
            self.msg_count += 1

    def increment_eml_count(self):
        with self.eml_count_lock:
            self.eml_count += 1
            
    def increment_mbox_count(self):
        with self.mbox_count_lock:
            self.mbox_count += 1
            
    def increment_pst_exception_count(self):
        with self.pst_exception_count_lock:
            self.pst_exception_count += 1

    def increment_msg_exception_count(self):
        with self.msg_exception_count_lock:
            self.msg_exception_count += 1

    def increment_eml_exception_count(self):
        with self.eml_exception_count_lock:
            self.eml_exception_count += 1
            
    def increment_mbox_exception_count(self):
        with self.mbox_exception_count_lock:
            self.mbox_exception_count += 1
            
    def increment_pst_index_exception_count(self):
        with self.pst_index_exception_count_lock:
            self.pst_index_exception_count += 1

    def increment_msg_index_exception_count(self):
        with self.msg_index_exception_count_lock:
            self.msg_index_exception_count += 1

    def increment_eml_index_exception_count(self):
        with self.eml_index_exception_count_lock:
            self.eml_index_exception_count += 1
            
    def increment_mbox_index_exception_count(self):
        with self.mbox_index_exception_count_lock:
            self.mbox_index_exception_count += 1
            
    def increment_pst_msg_count(self):
        with self.pst_msg_count_lock:
            self.pst_msg_count += 1
            
    def increment_pst_msg_exception_count(self):
        with self.pst_msg_exception_count_lock:
            self.pst_msg_exception_count += 1

    def increment_pst_msg_index_count(self):
        with self.pst_msg_index_count_lock:
            self.pst_msg_index_count += 1

    def increment_pst_msg_index_exception_count(self):
        with self.pst_msg_index_exception_count_lock:
            self.pst_msg_index_exception_count += 1
            
    def increment_eml_msg_count(self):
        with self.eml_msg_count_lock:
            self.eml_msg_count += 1
            
    def increment_eml_msg_exception_count(self):
        with self.eml_msg_exception_count_lock:
            self.eml_msg_exception_count += 1

    def increment_eml_msg_index_count(self):
        with self.eml_msg_index_count_lock:
            self.eml_msg_index_count += 1

    def increment_eml_msg_index_exception_count(self):
        with self.eml_msg_index_exception_count_lock:
            self.eml_msg_index_exception_count += 1 
                       
    def increment_mbox_msg_count(self):
        with self.mbox_msg_count_lock:
            self.mbox_msg_count += 1
            
    def increment_mbox_msg_exception_count(self):
        with self.mbox_msg_exception_count_lock:
            self.mbox_msg_exception_count += 1
            
    def increment_mbox_msg_index_count(self):
        with self.mbox_msg_index_count_lock:
            self.mbox_msg_index_count += 1

    def increment_mbox_msg_index_exception_count(self):
        with self.mbox_msg_index_exception_count_lock:
            self.mbox_msg_index_exception_count += 1
            
    def increment_attachd_file_count(self):
        with self.attached_file_count_lock:
            self.attached_file_count += 1
            
    def increment_total_count(self):
        with self.total_count_lock:
            self.total_count += 1

    def increment_compress_count(self):
        with self.compress_count_lock:
            self.compress_count += 1
            
    def increment_decompress_count(self):
        with self.decompress_count_lock:
            self.decompress_count += 1
            
    def increment_decompress_exception_count(self):
        with self.decompress_exception_count_lock:
            self.decompress_exception_count += 1
            
# class CountersWithLock:
#     def __init__(self):
#         self.files_count = AtomicInteger()  # AtomicInteger로 변경
#         self.files_count_lock = Lock()  # Lock 객체는 유지

#         # 나머지 변수들도 AtomicInteger로 변경하고 Lock 객체는 그대로 사용
#         self.files_exception_count = AtomicInteger()
#         self.files_without_extensions = AtomicInteger()
#         self.files_index_exception_count = AtomicInteger()
#         self.fileinfo_analyze_count = AtomicInteger()
        
#         self.increment_pst_count = AtomicInteger()
#         self.msg_count = AtomicInteger()
#         self.eml_count = AtomicInteger()
                
#         self.pst_exception_count = AtomicInteger()
#         self.msg_exception_count = AtomicInteger()
#         self.eml_exception_count = AtomicInteger()
        
#         self.pst_index_exception_count = AtomicInteger()
#         self.msg_index_exception_count = AtomicInteger()
#         self.eml_index_exception_count = AtomicInteger()  
        
#         self.increment_pst_msg_count = AtomicInteger()
#         self.pst_msg_index_count = AtomicInteger()
#         self.pst_msg_index_exception_count = AtomicInteger()                
        
#         self.total_count = AtomicInteger()
#         self.decompress_count = AtomicInteger()        

# CountersWithLock 클래스의 인스턴스 생성
# counters = CountersWithLock()
class MetaInfo:
    def __init__(self):
        self.doc_meta_info = {

        }

        self.mail_meta_info = {

        }

        self.pst_attachement = {

        }

        self.pst_mail_meta_info = {

        }
        



"""
파일 관련 정보 인덱스 구조
@timestamp
file : {
	'accessed' : 접근 시간,
	'created' : 생성 시간,
	'ctime' : 메타 변경 시간,
	'mtime' : 마지막 수정 시간,
	'directory' : 파일 경로,
	'name' : 파일 이름,
	'path' : 파일 전체 경로,
	'message' : 파일 내용,
	'summary' : 파일 요약(300자),
	'attributes' : 속성(ex. archive, compressed, directory, encrypted, execute, hidden, read, readoly, system, write, etc.. ),
	'owner' : 파일 owner 이름,
	'extension' : 확장자,
	'mime_type' : 컨텐츠 타입(ex. image/png),
	'size' : 파일 사이즈,
	'type' : 파일 유형(file, dir, symlink, etc.. ),
	'meta_info' : {
	
	}
},
'uuid': ' ',
'tags' : 태그(ex. ["file", "email"] )





이메일 관련 인덱스 구조
@timestamp
email: {
	'origin' : {
		'path' : '원본 메일 파일 전체 위치'
	},
	'name' : '메일 파일 이름',
	'path' : '메일 파일 전체 위치',
	'from' : {
		'address' : 헤더의 보낸 사람 주소
	},
	'sender' : {
		'address' : 메세지 실제 전송한 주소
	},
	'to' : {
		'address' : 수신자 이메일 주소
	},
	'cc' : {
		'address' : 참조 주소
	},
	'bcc' : {
		'address' : 숨은 참조 주소
	},
	'reply_to' : {
		'address' : 회신 주소
	},
	'x_mailer' : 이메일을 보낼 데 사용한 app,
	'message_id' : message_id,
	'subject' : 제목,
	'content_type' : MIME type,
	'delivery_timestamp' : 이메일 수신 날짜,
	'origination_timestamp' : 이메일 작성 날짜,
	'attachments' : {
		'file' : {
			'extension' : 확장자,
			'mime_type' : 컨텐츠 타입,
			'name' : 확장자를 포함한 파일 이름,
			'size' : 사이즈,
			'path' : ex)/data/attachment.txt
			}
	},
	'message' : 이메일 내용,
	'summary' : 이메일 내용(요약 300자),
},
'uuid' : uuid,
'tags' : 태그



"""