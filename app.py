from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import numpy as np
import os
import sys
import json
import re
import csv
import dashscope
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from google.cloud import storage
import datetime
import time
from datetime import timedelta


BUCKET_NAME = 'new-req-dedup.appspot.com'

def generate_signed_url(bucket_name, blob_name):
    """生成用于下载的签名 URL"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 生成一个有效期为1小时的签名 URL
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),  # 签名 URL 有效期
        method="GET"  # HTTP 请求类型
    )
    return url



# global counter for access log
global_access_log_id = 0


# 初始化 Cloud Storage 客户端
storage_client = storage.Client()

# upload files to gcs
def upload_to_gcs(file_path, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}.")

# upload json array
def save_json_to_gcs(data, destination_blob_name):
    """Saves deduplication results to GCS as a JSON file."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    print(f"Results saved to {destination_blob_name}.")

# load json from gcs
def load_json_from_gcs(bucket_name, blob_name):
    try:
        # 创建 GCS 客户端
        client = storage.Client()

        # 获取指定的 bucket
        bucket = client.get_bucket(bucket_name)

        # 获取指定的 blob（文件）
        blob = bucket.blob(blob_name)

        # 下载 blob 内容为字符串
        json_data = blob.download_as_text()

        # 将字符串转换为 Python 对象
        data = json.loads(json_data)

        return data

    except Exception as e:
        return None



# When server reboots, try to reload the previous counter
summary_blob_name = f"dedup_summary.json"
if global_access_log_id == 0:
    dedup_summary = load_json_from_gcs(BUCKET_NAME, summary_blob_name)
    if dedup_summary != None:
        global_access_log_id = dedup_summary['total_num']
        print(f'After reboot, retrieving previously saved global_access_log_id = {global_access_log_id}')


# write the access log
def log_access(log_id, ref_file, target_file, dedup_file,
               ref_count, query_count, dup_count, start_time, end_time):
    log_data = {
        "start_time": start_time,
        "end_time": end_time,
        "reference_file": ref_file,
        "target_file": target_file,
        "dedup_file": dedup_file,
        "reference_count": ref_count,
        "llm_query_count": query_count,
        "dup_count": dup_count,
        "id": log_id
    }

    log_file_name = f"access_{log_id}/access_log.json"
    save_json_to_gcs(log_data, log_file_name)



# comment out the following if need memory profiler
#import psutil
#from functools import wraps

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = '/tmp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# comment out the following if need memory profiler
__ = '''
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

def memory_profiler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_mem = get_memory_usage()
        print(f"Memory usage before {func.__name__}: {start_mem} MB")
        result = func(*args, **kwargs)
        end_mem = get_memory_usage()
        print(f"Memory usage after {func.__name__}: {end_mem} MB")
        return result
    return wrapper
'''

@app.route('/history')
def history():
    logs = []

    # 从GCP获取所有日志概要信息 (log_id 从1开始)
    log_id = 1
    while True:
        log_file_name = f"access_{log_id}/access_log.json"
        data = load_json_from_gcs(BUCKET_NAME, log_file_name)
        if data != None:
            logs.append(data)
            log_id += 1
        else:
            break

    print(f'In history(): logs = {logs}')
    return render_template('history.html', logs=logs)

@app.route('/history/<log_id>')
def history_detail(log_id):
    # 根据 log_id 获取日志概要信息
    log_file_name = f"access_{log_id}/access_log.json"
    log = load_json_from_gcs(BUCKET_NAME, log_file_name)
    # 根据 log_id 获取详细信息
    dedup_result_blob_name = f"access_{log_id}/dedup_results.json"
    results = load_json_from_gcs(BUCKET_NAME, dedup_result_blob_name)

    #ref_download_url = f"https://storage.googleapis.com/{BUCKET_NAME}/access_{log_id}/reference.csv"
    #target_download_url = f"https://storage.googleapis.com/{BUCKET_NAME}/access_{log_id}/target.csv"

    # 使用签名 URL
    ref_download_url = generate_signed_url(BUCKET_NAME, f"access_{log_id}/reference.csv")
    target_download_url = generate_signed_url(BUCKET_NAME, f"access_{log_id}/target.csv")

    return render_template('history_detail.html', log=log, results=results,
                           ref_download_url=ref_download_url, target_download_url=target_download_url)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory('downloads', filename, as_attachment=True)

ref_blob_name = ""
target_blob_name = ""
# Upload reference and target files
@app.route('/upload', methods=['POST'])
def upload_files():
    global global_access_log_id
    global ref_blob_name
    global target_blob_name

    if 'reference_file' not in request.files or 'target_file' not in request.files:
        return jsonify({'error': 'Both files are required.'})


    reference_file = request.files['reference_file']
    target_file = request.files['target_file']

    print(f'reference_file = {reference_file}')
    print(f'target_file = {target_file}')

    if reference_file and target_file:
        # increment global access log counter
        global_access_log_id += 1
        print(f'Start processing with global_access_log_id = {global_access_log_id}')

        reference_file_path = os.path.join(UPLOAD_FOLDER, 'reference.csv')
        target_file_path = os.path.join(UPLOAD_FOLDER, 'target.csv')


        print(f'reference_file_path = {reference_file_path}')
        print(f'target_file_path = {target_file_path}')


        # Save files to server, with always the same name, for dedup processing
        reference_file.save(reference_file_path)
        target_file.save(target_file_path)

        # 上传用户的输入文件到 GCS，用于历史记录
        ref_blob_name = f"access_{global_access_log_id}/"+reference_file.filename
        upload_to_gcs(reference_file_path, ref_blob_name)
        target_blob_name = f"access_{global_access_log_id}/"+target_file.filename
        upload_to_gcs(target_file_path, target_blob_name)


        return jsonify({'message': 'Files uploaded successfully.'})
    else:
        return jsonify({'error': 'File upload failed.'})

# 准备embedding model
def prepare_embedding_model(model_name):
    # Embed
    #model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    #model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f'calling prepare_embedding_model() with model_name: {model_name}')

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    try:
        #print(f"Start preparing the model ..., name = {model_name}")
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"Finished preparing the model ..., name = {model_name}, hf = {hf}")
    except Exception as e:
        print(f"Exception occurred in prepare_embedding_model(): {e}")
    
    return hf

# 输入两个字符串，基于embedding计算它们之间的余弦相似性
def calc_cosine_similarity(embedding_model, str1, str2):

    verbose = True
    if verbose:
        print(f'str1 = {str1}')
        print(f'str2 = {str2}')
        
    embed1 = embedding_model.embed_query(str1)
    embed2 = embedding_model.embed_query(str2)

    cosine_similarity = np.dot(embed1, embed2)

    return cosine_similarity

# 判断是否是中文字符
def is_chinese(string):
    # 正则表达式匹配中文字符
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(string))

# 构建prompt，用于向LLM发送实现去重
prompt_english = '''
    Please compare the following two requirements, with subject and description, and tell me whether they are very similar and should be duplicated.
    Please reply with the following format:
    * Probability: a number between 0% to 100%, showing how much you recommend to set the two tickets to duplicated
    * Analysis: Provide your detailed recommendation
    * New Requirement: If the probability is > 70%, draft a new requirement to combine the old two requirements, with
    ** Subject: <the new subject>
    ** Description: <the new description>
    '''
prompt_chinese = '''
    请对比以下两条需求的subject和description，判断它们是否非常相似因而可以判定为重复
    请按照以下格式回答：
    * 重复可能性: 请给出一个介于0% to 100%的数字，表示你建议将这两条需求设置为重复的程度
    * 分析: 请给出你的具体分析和建议
    * 新需求: 如果重复可能性 > 70%，请编写一条新的需求用于合并原有的两条需求，包括
    ** 标题: <新需求的标题>
    ** 描述: <新需求的描述>
'''
def form_query(use_chinese, req1__sub_and_desc, req2__sub_and_desc):
    
    # 有中文就用中文prompt，否则就用英文prompt
    if use_chinese:
        query_content = prompt_chinese
    else:
        query_content = prompt_english


    query_content = query_content + '\n\n-- Ticket1\n' + req1__sub_and_desc + '\n\n-- Ticket2\n' + req2__sub_and_desc

    return query_content

# 解析通义千问返回结果
def parse_rsp(use_chinese, text):
    # 使用正则表达式匹配, 允许匹配多行 (>0.5时，每段前面有'*')
    if use_chinese:
        pattern = r'\* 重复可能性:\s*(.*?)\s*\n\* 分析:\s*([\s\S]*?)\s*\n\* 新需求:\s*([\s\S]*)'
    else:
        pattern = r'\* Probability:\s*(.*?)\s*\n\* Analysis:\s*([\s\S]*?)\s*\n\* New Requirement:\s*([\s\S]*)'
    matches = re.search(pattern, text, re.MULTILINE)

    if not matches:
        # 使用正则表达式匹配, 允许匹配多行 (<=0.5时，每段前面没有'*')
        pattern = r'Probability:\s*(.*?)\s*\nAnalysis:\s*([\s\S]*?)\s*\nNew Requirement:\s*([\s\S]*)'
        matches = re.search(pattern, text, re.MULTILINE)

    if matches:
        return {
            'Probability': matches.group(1).strip(),
            'Analysis': matches.group(2).strip(),
            'New Requirement': matches.group(3).strip()
        }
    else:
        return {
            'Probability': None,
            'Analysis': None,
            'New Requirement': None
        }

# 调用通义千问大模型
from http import HTTPStatus
import dashscope
import os

def call_tongyi(use_chinese, query_content, stream_mode = False):
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    
    messages = [
        {'role': 'user', 'content': query_content}]


    rsp_str = ''
    
    if stream_mode:
        # streaming mode. If processing time is long, this can be used to get partial response when still processing
        try:
            responses = dashscope.Generation.call("qwen-max",
                                    messages=messages,
                                    result_format='message',  # set the result to be "message"  format.
                                    stream=True, # set streaming output
                                    incremental_output=True  # get streaming output incrementally
                                    )
        except Exception as e:
            print(f"Exception occurred in call_tongyi(): {e}")


        for response in responses:
            if response.status_code == HTTPStatus.OK:
                #print(response.output.choices[0]['message']['content'],end='')
                rsp_str = rsp_str + response.output.choices[0]['message']['content']
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
    else:
        # no need to use streaming mode. See whether this can save processing time
        try:
            response = dashscope.Generation.call("qwen-max",
                                        messages=messages,
                                        result_format='message'
                                        )
        except Exception as e:
            print(f"Exception occurred in call_tongyi(): {e}")


        if response.status_code == HTTPStatus.OK:
            #print(response.output.choices[0]['message']['content'],end='')
            rsp_str = response.output.choices[0]['message']['content']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    

    rsp_json_data = parse_rsp(use_chinese, rsp_str)

    #print(rsp_str, '\n\n')
    #print('----->>> Compared with parsed data ----->>> \n')
    print(f'Response: {rsp_json_data}')

    return rsp_json_data


# set embedding models as global, in order to save memmory usage
embedding_model_chinese = None
embedding_model_english = None

# Process deduplication
@app.route('/deduplicate', methods=['POST'])
#@memory_profiler
def deduplicate():
    global embedding_model_chinese
    global embedding_model_english
    global global_access_log_id

    start_time = str(datetime.datetime.now())

    reference_file_path = os.path.join(UPLOAD_FOLDER, 'reference.csv')
    target_file_path = os.path.join(UPLOAD_FOLDER, 'target.csv')

    print(f'reference_file_path = {reference_file_path}')
    print(f'target_file_path = {target_file_path}')


    if not os.path.exists(reference_file_path) or not os.path.exists(target_file_path):
        print(f'Error: in deduplicate(), ref or target file not found')
        return jsonify({'error': 'Files not found. Please upload files first.'})
    
    COSINE_SIMILARITY_THRESHOLD = 0.5
    LLM_PROBABILITY_THRESHOLD = 0.7
    MAX_LLM_QUERY_ALLOWED = 10 # 用户每次使用，最多允许调用LLM的次数（要花钱的）

    # Read target data
    target_data = []
    with open(target_file_path, 'r') as tgt_file:
        reader = csv.DictReader(tgt_file)
        target_data = list(reader)

    # 这里只比较目标需求的第一行
    target_id = str(target_data[0]['id'])
    target_subject = str(target_data[0]['subject'])
    target_description = str(target_data[0]['description'])
    print(f'target_id: {target_id}, target_subject:{target_subject}, target_description:{target_description}')

    ref_num = 0
    # Read reference and target CSV files
    reference_data = []
    with open(reference_file_path, 'r') as ref_file:
        reader = csv.DictReader(ref_file)
        for row in reader:
            reference_data.append(row)
            ref_num += 1

    print(f'ref_num = {ref_num}')


    dedup_results = []
    count = 0
    query_count = 0
    dup_count = 0

    # use multi-lingual model
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_model_chinese = embedding_model_english = prepare_embedding_model(model_name)

    # try to use multi lingual modal
    __ = '''
    if embedding_model_chinese == None:
        # chinese model for embedding
        #model_name_chinese = "shibing624/text2vec-base-chinese"
        #model_name_chinese = "nghuyong/ernie-3.0-base-zh"
        model_name_chinese = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        embedding_model_chinese = prepare_embedding_model(model_name_chinese)

    if embedding_model_english == None:
        # english model for embedding
        model_name_english = "sentence-transformers/all-mpnet-base-v2"
        embedding_model_english = prepare_embedding_model(model_name_english)
        '''


    for _, ref_row in enumerate(reference_data):
        reference_id = str(ref_row['id'])
        reference_subject = str(ref_row['subject'])
        reference_description = str(ref_row['description'])

        target__sub_and_desc = target_subject + '\n\n' + target_description
        ref__sub_and_desc = reference_subject + '\n\n' + reference_description

        if is_chinese(reference_subject) or is_chinese(target_subject):
            use_chinese = True
            print(f'====> using Chinese embedding ====>')
            cosine_similarity = calc_cosine_similarity(embedding_model_chinese, target__sub_and_desc, ref__sub_and_desc)
        else:
            use_chinese = False
            print(f'====> using English embedding ====>')
            cosine_similarity = calc_cosine_similarity(embedding_model_english, target__sub_and_desc, ref__sub_and_desc)
        

        print(f'Processing comparison NO. {count+1}: target_id = {target_id}, reference_id = {reference_id}')

        if (cosine_similarity < COSINE_SIMILARITY_THRESHOLD):
            probability = "0"
            analysis = "两条需求的余弦相似性为" + f"{cosine_similarity:.2f}" + " < 0.5，因此不调用大模型判断"
            new_requirement = "N.A."

            temp_prob = 0.0

        elif query_count < MAX_LLM_QUERY_ALLOWED:
            query_count += 1
            query = form_query(use_chinese, target__sub_and_desc, ref__sub_and_desc)
            rsp_json_data = call_tongyi(use_chinese, query, stream_mode = False)

            probability = rsp_json_data['Probability']
            analysis = rsp_json_data['Analysis']
            new_requirement = rsp_json_data['New Requirement']
            print(f'Processing query NO. {query_count}: target_id = {target_id}, reference_id = {reference_id}')
            print(f'rsp_json_data: {rsp_json_data}')

            temp_prob = float(probability.rstrip('%')) / 100

        else:
            probability = "-1"
            analysis = "两条需求的余弦相似性为" + f"{cosine_similarity:.2f}" + f" > 0.5，但LLM调用次数已超限:{MAX_LLM_QUERY_ALLOWED}次，不再调用（省钱）"
            new_requirement = "N.A."

            temp_prob = -1.0

        json_data = {
            "reference_id": str(ref_row['id']),
            "target_id": str(target_data[0]['id']),
            "cosine_similarity": f"{cosine_similarity:.2f}",
            "dup_probability": probability,
            "analysis": str(analysis),
            "suggestion": str(new_requirement)
        }
        if temp_prob > 0.7:
            dup_count += 1

        count += 1
        print(f'json_data = {json.dumps(json_data, indent=4)}')
        dedup_results.append(json_data)

    print(f'Totally processed {count} ref data rows, queried {query_count} times')



    # 保存去重结果到 GCS
    dedup_result_blob_name = f"access_{global_access_log_id}/dedup_results.json"
    save_json_to_gcs(dedup_results, dedup_result_blob_name)

    # 保存本次访问的概要信息到 GCS
    end_time = str(datetime.datetime.now())
    log_access(global_access_log_id,
        ref_blob_name,
        target_blob_name,
        dedup_result_blob_name,
        count,   # 总需求比对数量
        query_count,  # 调用LLM次数
        dup_count, # 判定重复次数
        start_time,
        end_time
    )

    dedup_summary = {"total_num" : global_access_log_id}
    save_json_to_gcs(dedup_summary, summary_blob_name)

    return jsonify(dedup_results)

if __name__ == '__main__':
    app.run(port = 5002, debug=True)

