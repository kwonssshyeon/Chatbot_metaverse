import torch 
import os
import pandas
from transformers import AutoModel,  AutoTokenizer, AutoModelForQuestionAnswering


model_ckpt = "timpal0l/mdeberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
 
question = input(">>")
context = """인사동한정식'인사동 한정식'은 전통 한정식 전문점입니다.
주소는 대구광역시 수성구 청수로24길 36-1 (두산동)입니다.
문의:053-761-2012
휴일은 일요일,명절휴무입니다.
수변 산책과 분수쇼, 야경을 한 자리에서 즐기기 ‣ \
둘레 2km의 수성못을 따라 바늘꽃, 연꽃, 갈대 등이 어우러진 수변 데크 로드와 울창한 왕벚나무, 버드나무 가로수길이 펼쳐진다. \
밤에는 물 위에 비친 조명이 물감을 풀어 놓은 듯 아름답게 반짝인다. 하루 4회 영상음악분수가 가동되어 볼거리를 더한다. \
대구를 대표하는 호수 공원으로 아름다운 산책 코스와 분수쇼 등의 야경 명소로 유명하며 다양한 맛집과 카페, \
오리배 타기 및 수성랜드 등 즐길 거리가 넘쳐 나들이 혹은 데이트코스로 많이 사랑받고 있습니다.\
금수강산해물탕
산지직송의 신선한 재료만을 사용하여 시원하고 얼큰한 맛이 특징
주소 : 대구광역시 수성구 들안로 60 (두산동)
문의 : 053-766-9092
휴일 : 연중무휴"""

inputs = tokenizer(question, context, return_tensors='pt')

model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

with torch.no_grad():
    outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits)+1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)

print(f"질문 : {question}")
print(f"답변 : {answer}")