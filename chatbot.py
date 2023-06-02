import os
import torch
import pandas
from transformers import pipeline, AutoModelForCausalLM, AutoModel,  AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util


GENERATE_MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'
generate_tokenizer = AutoTokenizer.from_pretrained(GENERATE_MODEL)

QA_MODEL = "timpal0l/mdeberta-v3-base-squad2"
QA_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)

sentence_transformer_path="KDHyun08/TAACO_STS"
embedder = SentenceTransformer(sentence_transformer_path)


model = AutoModelForCausalLM.from_pretrained(
    GENERATE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline(
    'text-generation', 
    model=model,
    tokenizer=GENERATE_MODEL,
    device=0
)


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Input(입력):\n{input}\n\n### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
    ),
}



def data_prep_from_txt(file_name : str, o_data : dict = None) :
    '''
    Here we assume the data is a text file and follows the format :
    
    ■ location 
    
    
    [데이터]
    summary....
    
    [질의응답]
    Q1: question 1
    A1: answer1 
    .
    .
    .
    
    '''
    with open(file_name,'r') as fil :
        data=fil.read()
    data=data.replace('n\n\n\n\n\n\n\n\n\n','')
    data=data.split("■")

    data=[data[i].split('[질의응답]') for i in range(len(data))]
    for i in range(len(data)) :
        try :
            data[i][1]=data[i][1].split('\nQ')
        except Exception as e  :
            continue
    for i in range(len(data)) :
        try :
            for j in range(len(data[i][1])) :
                data[i][1][j]=data[i][1][j].split('\nA')
        except Exception as e  :
            continue
    data=data[1:]

    for i in range(len(data)) :
        data[i][1]=data[i][1][1:]
    
    dataf = {} if not o_data else o_data
    for i in range(len(data)):
        dataf[data[i][0].split('\n')[0][1:]] = {
               'context' : data[i][0].split('[데이터]')[1] ,
               'question' : [data[i][1][j][0].split(':')[1].replace('\n','')  for j in range(len(data[i][1]))],
               'answer' : [data[i][1][j][1].split(':')[1].replace('\n','')  for j in range(len(data[i][1]))],
                }  
        
    return dataf


def get_category(question : str, kD : dict) : #카테고리를 반환
    corpus = list(kD.keys())
    hits = util.semantic_search(embedder.encode(question),embedder.encode(corpus))

    #유사도가 0.5이상이면 hits내용(카테고리), 이하면 빈 문자열 반환
    category = corpus[hits[0][0]['corpus_id']] if hits[0][0]['score'] > 0.5 else ''
    return category




def get_answer(data,question) :     #입력질문과 가장 비슷한 질문 1개의 답변 반환
    hits=util.semantic_search(embedder.encode(question),embedder.encode(data['question']))
    l = len(hits[0])

    return (data['answer'][hits[0][0]['corpus_id']],hits[0][0]['score'])  



def get_context(data,question,size,tokenizer): #(해당카테고리 데이터셋 전체, 입력질문, 본문 길이, 토크나이저) context반환
    hits=util.semantic_search(embedder.encode(question),embedder.encode(data['context'].split('\n\n'))) #질문과 카테고리 본문
    context = data['context']
    #print(hits)
    if(hits[0][0]['score']<0.4):
        return ""
    length = 5
    while(1):
        tok_size = len(context)
        if(tok_size<=size):
            break
        context='\n\n'.join([data['context'].split('\n\n')[hits[0][i]['corpus_id']] for i in range(length)])
        length-=1

    return context



def answer_in_context(question, context):
    context = '"""' + context +  '"""'
    
    inputs = QA_tokenizer(question, context, return_tensors='pt')
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)+1
    answer_span = inputs["input_ids"][0][start_idx:end_idx]
    answer = QA_tokenizer.decode(answer_span)

    return answer


def generate_answer(x, context='', is_input_full=False):
    ans = pipe(
        f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:", 
        do_sample=True, 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        pad_token_id=2,
        eos_token_id=2,
    )
    return ans[0]['generated_text']



def make_convo( model,tokenizer, kd : dict) :
    
    #input이 없는 경우
    hist_no_input="Below is an instruction that describes a task.\n\
            아래는 작업을 설명하는 명령어입니다.\n\n\
            Write a response that appropriately completes the request.\n\
            명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
    
    #input이 있는 경우
    hist_with_input="Below is an instruction that describes a task, paired with an input that provides further context.\n\
        아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n\
            Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n\
                ### Input(입력):\n"

    
    category=''
    QA_hist=''
    user_input=''
    context=''
    i=0
    prompt=''

    while True :
        find_answer=False
        
        user_input=input('>>') 
        if user_input.lower() == 'bye' :
            break
        
        current_category= get_category(user_input,kd)
 

        #이전 카테고리를 기억하고 카테고리가 바뀔때마다 히스토리를 초기화
        if(category=="" and current_category!=""):
            category = current_category
        elif(current_category != category and current_category!=""):
            category = current_category
            QA_hist = ""

        
        if(category!=""):
            matching_answer = get_answer(data=kd[category],question=user_input)

            #카테고리 o 질문 o (카테고리 x 질문 o)
            if(matching_answer[1] > 0.9):
                answer = matching_answer[0]
                find_answer = True

            #카테고리 o 질문 x
            else:
                context = get_context(kd[category],user_input,512-64,tokenizer)
                if(context!=""):
                    answer = answer_in_context(user_input,context)
                    

                if(context=="" or answer == "[CLS]"):
                    prompt = hist_with_input+"#####\nPrevious_chat\n\n"+QA_hist+"\n#####\n"+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):" 
                    answer = generate_answer(user_input,prompt)
                find_answer=True

        
        #카테고리 x 질문 x
        if(current_category=="" and (not find_answer)) :
            answer = generate_answer(user_input)


        QA_hist=QA_hist+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):\n"+answer
        i=i+1

        print(f'bot: {answer}')





if __name__=='__main__' :
    
    path_dir='data' #replace with the path to the txt data files
    df={}
    for fil in os.listdir(path_dir) :
        if not fil.endswith('txt') :
            continue
        df=data_prep_from_txt(path_dir+'/'+fil,df)

    
    
    make_convo(model,generate_tokenizer,df)
