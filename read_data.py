import numpy as np
import linecache
import json
import re
import nltk

def clean_str(string):
	string = string.strip().lower()

	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r',', '', string)
	string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)
	if 'let\'s' in string:
	    string = re.sub(r'let\'s', 'let us', string)
	if 'lets' in string:
	    string = re.sub(r'lets', 'let us', string)

	string = re.sub(r"\'s", " is", string)
	string = re.sub(r"\'ve", " have", string)
	if 'wont ' in string:
	    string = re.sub(r"won\'?t", "will not", string)
	if 'won\'t ' in string:
	    string = re.sub(r"won\'?t", "will not", string)

	if 'cant ' in string:
	    string = re.sub(r"n\'?t", " can not", string)
	if 'can\'t ' in string:
	    string = re.sub(r"n\'?t", " can not", string)

	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r",", "", string)
	string = re.sub(r"!", "", string)
	string = re.sub(r"\(", "", string)
	string = re.sub(r"\)", "", string)
	string = re.sub(r"\?", "", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"\'", '', string)

	return string.strip()
def preprocess(sentence):
	sentence = sentence.lower()
	sentence=clean_str(sentence)
	return sentence



corpus_data=[]
for i in range(1,10000):
	corpus_data.append(linecache.getline('rare_entity/corpus.txt',i))
	#print(corpus_data[i])
	#print('--------------------')
entities=[]

f=open('rare_entity/entities.txt','r')
entities=f.readlines()
c=0
dic={}
dictionary={}
dictionary['<blank>']=0
for e in entities:
	temp_e=e.split('\t')
	sentence=preprocess(temp_e[4].split('.')[0])
	label=temp_e[1]
	dic[temp_e[0]]=(label,sentence)
	words=sentence.split()
	for w in words:
		if w not in dictionary:
			dictionary[w]=len(dictionary)

sentences=[]


max_div=0
max_ents=0
max_ents_len=0
for c in corpus_data:
	sent=preprocess(c)
	s=sent.split()
	
	ent=[]
	s1=''
	for k in s:
		if re.match('^9202[a-z]',k):
			s1+='<blank> '
			ent.append(dic[k])
			if max_ents_len<len(dic[k][1].split()):
				max_ents_len=len(dic[k][1].split())
		else:
			if k not in dictionary:
				dictionary[k]=len(dictionary)
			s1+=k+' '
	s1=s1[:-1]
	o=s1.split()
	blank_indices=[]
	dot_indices=[]
	dic_blnk_dot={}
	i=0
	for k in o:
		if k == '<blank>':
			blank_indices.append(i)
			dic_blnk_dot[i]='<blank>'
		elif k == '.':
			dot_indices.append(i)
			dic_blnk_dot[i]='.'
		i+=1
	merged=blank_indices+dot_indices
	merged.sort()

	div_sentences=[]
	div_entities=[]
	include=True
	prev_c=0
	c2=0
	#blank dot blank dot dot dot
	#print(s1)
	while c2<len(merged):
		while c2<len(merged) and dic_blnk_dot[merged[c2]] is not '<blank>':
			c2+=1
		if c2+1<len(merged) and dic_blnk_dot[merged[c2+1]] is '<blank>':
			include=False
			break
		else:
			if c2==len(merged) or c2+1 ==len(merged):
				sent1=o[prev_c:]
			else:
				c3=c2+2
				for z in range(c3,len(merged)):
					if dic_blnk_dot[merged[z]]=='.':
						c3+=1
					else:
						break
				if c3==len(merged):
					sent1=o[prev_c:]
					if len(sent1)>2000:
						include=False
						break
					div_sentences.append(sent1)
					break
				else:
					sent1=o[prev_c:merged[c2+1]+1]
				
				prev_c=merged[c2+1]+1
			if len(sent1)>2000:
				include=False
				break
			
			#print('+++++++++++')
			div_sentences.append(sent1)
			
			c2+=1



	#print(len(div_sentences))
	
	if include:
		if len(div_sentences)>max_div:
			max_div=len(div_sentences)
		if max_ents<len(ent):
			max_ents=len(ent)
		sentences.append((div_sentences,ent))
print(max_ents)
print(max_ents_len)
print(max_div)
import pickle
with open('data.pickle','wb') as f:
	pickle.dump(sentences,f)
with open('dict.pickle','wb') as f:
	pickle.dump(dictionary,f)
