import os
import torch
import pandas
from transformers import pipeline, AutoModelForCausalLM, AutoModel,  AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util


GENERATE_MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'
generate_tokenizer = AutoTokenizer.from_pretrained(GENERATE_MODEL)
generate_tokenizer.pad_token_id = generate_tokenizer.eos_token_id
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


def ask(x, context='', is_input_full=False):
    ans = pipe(
        f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:", 
        do_sample=True, 
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    #print(ans[0]['generated_text'])
    return ans[0]['generated_text']


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


def get_context(question : str, kD : dict) : #카테고리를 반환
    global category
    corpus = list(kD.keys())
    hits = util.semantic_search(embedder.encode(question),embedder.encode(corpus))
    print("카테고리 hits: ", hits)

    #유사도가 0.5이상이면 hits내용(카테고리), 이하면 빈 문자열 반환
    category = corpus[hits[0][0]['corpus_id']] if hits[0][0]['score'] > 0.5 else ''
    return category


def get_answer(data,question) :     #data['question']에서 입력질문과 가장 비슷한 거 반환
    hits=util.semantic_search(embedder.encode(question),embedder.encode(data['question']))
    l = len(hits[0])
    print("get_ans함수의 hit 내용: ",hits)
    return (data['answer'][hits[0][0]['corpus_id']],hits[0][0]['score'])  


def get_fitting_mod(data,question,size,tokenizer,hist,prev) :   #일치하는 질문을 찾지 못했을때    
    #(해당카테고리 데이터셋 전체, 입력질문, 본문 길이, 토크나이저, 질문기록, 바로 전 답변)
    #print(data['context'].split('\n\n')) 해당 카테고리의 context내용(단락별로 리스트)
    
    hits=util.semantic_search(embedder.encode(question),embedder.encode(data['context'].split('\n\n'))) #질문과 카테고리 본문
    hitsp=util.semantic_search(embedder.encode(prev),embedder.encode(data['context'].split('\n\n')))    #이전 답변과 카테고리 본문
    x = PROMPT_DICT['prompt_input'].format(instruction=question, input=data['context']+hist)            #프롬프트에 instruction(질문),input(본문+질의응답 히스토리)
    print("x의 내용 출력: ",x)
    print("prev의 내용 출력:\n",prev)
    print("hist의 내용 출력: \n",hist)
    print("hits의 내용 출력: \n",hits)    #context의 어느 단락이랑 가장 유사한지
    print("hitsp의 내용 출력: \n",hitsp)      #이전질문은 context의 어느 단락이랑 가장 유사한지
    if hits[0][0]['score']<0.4 :
        return ''       #유사한 단락이 없으면 빈문자열
    length = len(hits[0])    #단락 개수
    while (1):
        tok_size =  tokenizer.encode(x,return_tensors="pt").shape[1]
        print("size: ",tok_size,"\n")
        if(tok_size<=size):     #size = 1048-128
            break

        print("------------------------------------")
        print("유사한 단락")
        print(data['context'].split('\n\n')[hitsp[0][0]['corpus_id']])
        
        length-=1
        #input은 이전답변과 유사도가 가장 높은 본문 + 현재 질문과 유사도가 높은 본문 순서대로
        x=PROMPT_DICT['prompt_input'].format(instruction=question, 
                                             input=data['context'].split('\n\n')[hitsp[0][0]['corpus_id']]+'\n\n'+'\n\n'.join([data['context'].split('\n\n')[hits[0][i]['corpus_id']] for i in range(length)])+'\n\n'+hist)
        print("[",length,"(l)]\n",x)
    return x

def find_in_context(question, context):

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
    print("답변: ",answer)
    return answer



def make_convo( model,tokenizer, kd : dict) :
    global prev_quest
    category=''

    #input이 없는 경우
    hist1="Below is an instruction that describes a task.\n\
            아래는 작업을 설명하는 명령어입니다.\n\n\
            Write a response that appropriately completes the request.\n\
            명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
    
    #input이 있는 경우
    hist2="Below is an instruction that describes a task, paired with an input that provides further context.\n\
        아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n\
            Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n\
                ### Input(입력):\n"
    QA_hist=''
    user_input=''
    context=''
    tok=False
    i=0
    x=('',0)
    da=''
    while True :
        user_input=input('>>') 
        if user_input.lower() == 'bye' :
            break
        n_context= get_context(user_input,kd)
        print("n_context 내용: ",n_context)
        if (n_context != category) and n_context!='' :   #카테고리가 이전질문과 같지 않을때 (QA_hist, 카테고리 기록 초기화)
            #2개 정도로 늘려서 다시 테스트해보기
            context=''
            QA_hist='' #질문-응답 쌍
            i=0
        if n_context!='' :          #일치하는 카테고리가 있을 때
            context=n_context       #-->이전의 카테고리를 기억하기 위함
        if context != '' :
            x=get_answer(data=kd[context],question=user_input)
            print("make_conv함수의 x내용: ",x)      #매치되는 (answer,정확도)
        if i==0:        #첫번째 질문이고 input이 없음
            model_inp=hist1+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):" 
            nhist=QA_hist+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):"
        else :          #첫번째 질문이 아니고 input이 있음
            model_inp=hist2+"#####\nPrevious_chat\n\n"+QA_hist+"\n#####\n"+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):" 
            nhist=QA_hist+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):"

        if x[1]>0.5 :           #반환된 답의 정확도가 0.5이상일 경우
            gen_text=x[0]
        
        
        else :  #x[1]<=0.5
            if context != '' :
                #if first query     
                if prev_quest=='':
                    print("bot: 해당 질문에 답을 할 수 없습니다. 감사합니다.")
                    continue
                #카테고리는 있지만, answer는 없을때
                da=get_fitting_mod(kd[context],user_input,512-64,tokenizer,QA_hist,prev_quest)    #(해당카테고리 데이터셋 전체, 입력질문, 본문 길이, 토크나이저, 질문기록, 바로 전 답변)
                print("da 내용 : ",da)
            mi=model_inp if da=='' else da    # --> mi = da  
            input_ids=tokenizer.encode(mi, return_tensors="pt").to(model.device)    #peft 모델로 답변 생성
            gen_text = ask(user_input,mi)
            # gen_tokens = model.generate(
            #     input_ids=input_ids, 
            #     max_new_tokens=128, #생성 토큰의 개수
            #     num_return_sequences=1, 
            #     temperature=0.5,        #이전에 나왔던 토큰의 확률을 0으로 바꿈 --> 반복적인 생성 제거
            #     no_repeat_ngram_size=6,
            #     do_sample=False,     #할때마다 다른 응답이 나오게 됨
            #     top_k=50,
            #     top_p=0.90,
            # )
            #gen_text = tokenizer.decode(gen_tokens[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        QA_hist=nhist+gen_text
        prev_quest=gen_text
        i=i+1
        category = n_context
        print(f'bot: {gen_text}')



if __name__=='__main__' :
    prev_quest=''
    category=''
    path_dir='data' #replace with the path to the txt data files
    df={}
    for fil in os.listdir(path_dir) :
        if not fil.endswith('txt') :
            continue
        df=data_prep_from_txt(path_dir+'/'+fil,df)


    make_convo(model,generate_tokenizer,df)

    # print(df)
    # while(1):
    #     user_input = input(">>")

    #     if user_input == "bye":
    #         break
    #     ask(user_input,df['수성못']['context'][:1000])