import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
with open('dict.pickle','rb') as f:
	dictionary=pickle.load(f)
num_epochs=20
batch_size=1

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
vocab_size=len(dictionary)+1
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs('content/glove.6B.300d.txt')
embed_matrix1 = np.zeros((vocab_size, 300),dtype=np.float32)#contains glove vectors for each word in vocabulary
for word,index in dictionary.items():
	try:
		embed_matrix1[index, :] = word_to_vec_map[word]
	except:
		embed_matrix1[index, :] = np.random.uniform(-1, 1, 300)
data=tf.placeholder(tf.float32,[batch_size,None,None,None])
data_seq_len_main=tf.placeholder(tf.float32,[batch_size,None,None])
data2dseqlen=tf.placeholder(tf.int32,[batch_size])
data_ent_des=tf.placeholder(tf.float32,[batch_size,None,None])
data_ent_seq_len_main=tf.placeholder(tf.float32,[batch_size,None])
position_of_blank=tf.placeholder(tf.int32,[batch_size,None,None])
with tf.name_scope('embed'):
	W = tf.get_variable(name = 'W', shape = embed_matrix1.shape, initializer = tf.constant_initializer(embed_matrix1), trainable = True)#embed_matrix1 is the pretrained glove embeddings

def embedding(inp):

	x=tf.cast(inp,tf.int32)
	embeddings_out = tf.nn.embedding_lookup(W,x)
	return embeddings_out
def data_emb1(ip):

	embed=tf.map_fn(embedding,ip)
	return embed
data_emb=tf.map_fn(data_emb1,data)
data_ent_des_emb=tf.map_fn(embedding,data_ent_des)
def encode_entities(ip):
	with tf.variable_scope("rnn1"):
		rnn_layers1 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size),output_keep_prob=0.5) for size in [128, 300]]
		multi_rnn_cell1 = tf.nn.rnn_cell.MultiRNNCell(rnn_layers1)
		outputs1, state1 = tf.nn.dynamic_rnn(cell=multi_rnn_cell1,inputs=ip[0],sequence_length=ip[1],dtype=tf.float32)
		return outputs1,state1[1].h
entity_encoder=tf.map_fn(encode_entities,(data_ent_des_emb,data_ent_seq_len_main),dtype=(tf.float32,tf.float32))
entity_encoder=entity_encoder[1]
entity_encoder=tf.expand_dims(entity_encoder,1)
entity_encoder=tf.tile(entity_encoder,[1,tf.shape(data)[1],1,1])
def pad(ip):
	def padin(ip1):
		zeros1=tf.zeros([300,ip1[1]],dtype=tf.float32)
		zeros2=tf.zeros([300,tf.shape(data)[3]-ip1[1]-1],dtype=tf.float32)
		ip2=tf.reshape(ip1[0],[-1,1])
		op=tf.concat((zeros1,ip2,zeros2),1)
		op=tf.transpose(op)
		return op,op
	padded=tf.map_fn(padin,(ip[0],ip[1]))
	padded=padded[0]
	return padded,padded
def outer(ip):
	out=tf.map_fn(pad,(ip[0],ip[1]))
	return out
final_pad=tf.map_fn(outer,(entity_encoder,position_of_blank),dtype=(tf.float32,tf.float32))
final_pad=final_pad[0]
data_emb=tf.add(data_emb,final_pad)

def herarchical(ip):
	def diff_each_blank(ip1):
			with tf.variable_scope("rnn2"):
				rnn_layers1 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size),output_keep_prob=0.5) for size in [128, 300]]
				multi_rnn_cell1 = tf.nn.rnn_cell.MultiRNNCell(rnn_layers1)
				outputs1, state1 = tf.nn.dynamic_rnn(cell=multi_rnn_cell1,inputs=ip1[0],sequence_length=ip1[1],dtype=tf.float32)
				return outputs1,state1[1].h
	each_op=tf.map_fn(diff_each_blank,(ip[0],ip[1]))
	each_op1=tf.reduce_mean(each_op[1],0)
	return each_op[1],each_op1
	
herarchical_lstm_op=tf.map_fn(herarchical,(data_emb,data_seq_len_main),dtype=(tf.float32,tf.float32))
herarchical_lstm1=herarchical_lstm_op[0]
herarchical_lstm2=herarchical_lstm_op[1]

with tf.variable_scope("rnn3"):
	rnn_layers2 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size),output_keep_prob=0.5) for size in [128, 300]]
	multi_rnn_cell2 = tf.nn.rnn_cell.MultiRNNCell(rnn_layers2)
	outputs2, state2 = tf.nn.dynamic_rnn(cell=multi_rnn_cell2,inputs=herarchical_lstm2,sequence_length=data2dseqlen,dtype=tf.float32)
lables=tf.placeholder(tf.float32,shape=[None,None,None])
w=tf.Variable(tf.truncated_normal([300,300]))
v=tf.Variable(tf.truncated_normal([300,1]))
b=tf.Variable(tf.constant(1.0,shape=[1]))

def loss(ip):
	def loss2(ip1):
		def loss3(ip2):
			expip2=tf.reshape(ip2[1],[300,1])
			#print(ip2[0])
			expip1=tf.reshape(ip2[0],[1,300])
			mul=tf.matmul(expip1,tf.matmul(w,expip2))
			return mul,mul
		mul=tf.map_fn(loss3,(ip1[0][0],ip1[0][1]))
		mul=mul[0]
		mul=tf.reshape(mul,[1,-1])
		mul2=tf.matmul(tf.reshape(ip1[0][2],[1,-1]),v)
		add1=tf.add(mul,tf.add(mul2,b))
		loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=add1,labels=ip1[1])
		add1=tf.nn.softmax(add1)
		return ((loss,loss,loss),add1)
	op=tf.map_fn(loss2,(ip,lables))
	return (op[0][0],op[0][0],op[1])



loss_fn1=tf.map_fn(loss,(herarchical_lstm1,entity_encoder,outputs2))
loss_fn=loss_fn1[0]
logits=loss_fn1[2]
loss_fn=tf.reduce_mean(loss_fn,0)
loss_fn=tf.reduce_mean(loss_fn)
opt=tf.train.AdamOptimizer(1e-3).minimize(loss_fn)
saver=tf.train.Saver()




	

sentences=[]
with open('data.pickle','rb') as f:
	sentences=pickle.load(f)
print(len(sentences))
train_sentences,test_sentences=sentences[:550],sentences[550:600]



import math
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
lossdic={}
train_acc=0
test_acc=0
stats=[]
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	for abcd in range(num_epochs):
		num_batches=math.floor(len(train_sentences)/batch_size)
		l=0
		train_acc=0
		train_samples=0
		for i in tqdm(range(num_batches)):
			cur_batch=train_sentences[i*batch_size:i*batch_size+batch_size]
			sent=np.zeros([batch_size,9,9,2000])
			blankposition=np.zeros([batch_size,9,9])
			seqlenmain=np.zeros([batch_size,9,9])
			seqlen2d=np.zeros([batch_size])
			entitiy_descriptions=np.zeros([batch_size,9,137])
			
			ent_seq_len_main=np.zeros([batch_size,9],dtype=np.float32)
			labs=np.zeros([batch_size,9,9])
			c1=0
			for c in cur_batch:
				sents=c[0]
				entities=c[1]
				
				c2=0
				seqlen2d[c1]=len(sents)
				for s in sents:
				
					c3=0
					#print(len(s))
					for k in sents:
						seqlenmain[c1,c2,c3]=len(s)
						c6=0

						for w in s:
							if w == '<blank>':
								blankposition[c1,c2,c3]=c6
							sent[c1,c2,c3,c6]=dictionary[w]
							c6+=1
						c3+=1
					c2+=1
				
				c4=0
				ent_seq_len_main[c1]=len(entities)
				for e1 in entities:
					e=e1[1]
					wd=e.split()
					c5=0
					labs[c1,c4,c4]=1
					ent_seq_len_main[c1,c4]=len(wd)
					for w in wd:
						entitiy_descriptions[c1,c4,c5]=dictionary[w]
						c5+=1
					c4+=1
				c2+=1
				c1+=1
			loss,_,lgts=sess.run([loss_fn,opt,logits],feed_dict={data2dseqlen:seqlen2d,position_of_blank:blankposition,data:sent,data_seq_len_main:seqlenmain,data_ent_des:entitiy_descriptions,data_ent_seq_len_main:ent_seq_len_main,lables:labs})
			l+=loss/num_batches
			
			for j in range(lgts.shape[0]):
				for z in range(lgts.shape[1]):
					t1=np.argmax(lgts[j,z,:])
					t2=np.argmax(labs[j,z,:])
					train_samples+=1
					if t1==t2:
						train_acc+=1
		print('train_acc:'+str(train_acc)+'/'+str(train_samples))

		
		print(abcd,' ',l)
		num_test_batches=math.floor(len(test_sentences)/batch_size)

		test_acc=0
		test_samples=0
		for i in tqdm(range(num_test_batches)):
			cur_batch=test_sentences[i*batch_size:i*batch_size+batch_size]
			sent=np.zeros([batch_size,9,9,2000])
			blankposition=np.zeros([batch_size,9,9])
			seqlenmain=np.zeros([batch_size,9,9])
			seqlen2d=np.zeros([batch_size])
			entitiy_descriptions=np.zeros([batch_size,9,137])
			
			ent_seq_len_main=np.zeros([batch_size,9],dtype=np.float32)
			labs=np.zeros([batch_size,9,9])
			c1=0
			for c in cur_batch:
				sents=c[0]
				entities=c[1]
				
				c2=0
				seqlen2d[c1]=len(sents)
				for s in sents:
				
					c3=0
					#print(len(s))
					for k in sents:
						seqlenmain[c1,c2,c3]=len(s)
						c6=0

						for w in s:
							if w == '<blank>':
								blankposition[c1,c2,c3]=c6
							sent[c1,c2,c3,c6]=dictionary[w]
							c6+=1
						c3+=1
					c2+=1
				
				c4=0
				ent_seq_len_main[c1]=len(entities)
				for e1 in entities:
					e=e1[1]
					wd=e.split()
					c5=0
					labs[c1,c4,c4]=1
					ent_seq_len_main[c1,c4]=len(wd)
					for w in wd:
						entitiy_descriptions[c1,c4,c5]=dictionary[w]
						c5+=1
					c4+=1
				c2+=1
				c1+=1
			lgts=sess.run(logits,feed_dict={data2dseqlen:seqlen2d,position_of_blank:blankposition,data:sent,data_seq_len_main:seqlenmain,data_ent_des:entitiy_descriptions,data_ent_seq_len_main:ent_seq_len_main,lables:labs})
			
			
			for j in range(lgts.shape[0]):
				for z in range(lgts.shape[1]):
					t1=np.argmax(lgts[j,z,:])
					t2=np.argmax(labs[j,z,:])
					test_samples+=1
					if t1==t2:
						test_acc+=1
		print('test acc:'+str(test_acc)+'/'+str(test_samples))
		try:
			os.mkdir('drive/runs/models/model_'+str(abcd))
		except:
			pass
		#saver.save(sess,'drive/runs/models/model_'+str(abcd)+'/entity_model')
		stats.append([float(test_acc)/float(test_samples),float(train_acc)/float(train_samples),l])
		with open('drive/stat.pickle','wb') as t:
			pickle.dump(stats,t)
		print('-------')




