#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[1]:


get_ipython().system('unzip data/data25959/归档.zip')


# #建立字典 还原数据

# In[2]:


line= open ('dict.txt').read()
line=line[1:]
line=line[:-2]
dict1 = {}
ones= line.split(',')
for one in ones:
    one=one.strip()
    try:        
        dict1[one.split(':')[0].replace("'","")]=int(one.split(':')[1])
    except IndexError as s:
        pass


# In[3]:


dict1_rev = {v:k for k,v in dict1.items()}
dict1_rev[1466]=','


# In[36]:


import pandas as pd
data=[]
with open('shuffle_Train_IDs.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        left,right=line.strip().split('\t')
        tmp_sents = []
        tmp=''
        tmp_sents.append(right)
        for word in left.strip().split(','):
            tmp=tmp+dict1_rev[int(word)]
        tmp_sents.append(tmp)
        data.append(tmp_sents)
df=pd.DataFrame(data,columns=['label','text_a'])
df.to_csv('all_data.tsv',sep='\t',index=False)   


# 数据增强：删除+替换

# In[49]:


df=pd.read_csv('pseudo_7.tsv',sep='\t')
print(len(df))


# In[8]:


m='0123456789'
new=[]
import random
for i in range(len(df)):
    temp=[]
    left=df['label'][i]
    right=df['text_a'][i]
    for j in right:
        if j in m:
            right=right.replace(j,'')
        else:
            continue
    if len(right)>2:
        u=random.randint(0,len(right)-2)
    if len(right)>4:
        v=random.randint(0,len(right)-4)
    if len(right)>6:
        w=random.randint(0,len(right)-6)
    if len(right)>8:
        x=random.randint(0,len(right)-8)
    if len(right)<30:
        right=right.replace(right[u:u+2],'')
    elif len(right)<50 and len(right)>30:
        right=right.replace(right[u:u+2],'')
        right=right.replace(right[v:v+2],'')
    elif len(right)<70 and len(right)>50:
        right=right.replace(right[u:u+2],'')
        right=right.replace(right[v:v+2],'')
        right=right.replace(right[w:w+2],'')
    else:
        right=right.replace(right[u:u+2],'')
        right=right.replace(right[v:v+2],'')
        right=right.replace(right[w:w+2],'')
        right=right.replace(right[x:x+2],'')
    temp.append(left)
    temp.append(right)
    new.append(temp)
df1=pd.DataFrame(new,columns=['label','text_a'])
print(df1[:5])
print(len(df1))    


# In[50]:


m='0123456789'
new1=[]
import random
for i in range(len(df)):
    temp=[]
    left=df['label'][i]
    right=df['text_a'][i]
    for j in right:
        if j in m:
            right=right.replace(j,'')
        else:
            continue
    if len(right)>2:
        u=random.randint(0,len(right)-2)
    if len(right)>4:
        v=random.randint(0,len(right)-4)
    if len(right)>6:
        w=random.randint(0,len(right)-6)
    if len(right)>8:
        x=random.randint(0,len(right)-8)
    if len(right)<30:
        right=right.replace(right[u:u+2],'')
    elif len(right)<50 and len(right)>30:
        right=right.replace(right[u:u+2],'')
        right=right.replace(right[v:v+2],'')
    elif len(right)<70 and len(right)>50:
        right=right.replace(right[u:u+2],'')
        right=right.replace(right[v:v+2],'')
        right=right.replace(right[w:w+2],'')
    else:
        right=right.replace(right[u:u+2],'')
        right=right.replace(right[v:v+2],'')
        right=right.replace(right[w:w+2],'')
        right=right.replace(right[x:x+2],'')
    temp.append(left)
    temp.append(right)
    new1.append(temp)
df2=pd.DataFrame(new1,columns=['label','text_a'])
print(df2[:5])
print(len(df2))


# In[9]:


m='0123456789'
new2=[]
import random
for i in range(len(df)):
    temp=[]
    left=df['label'][i]
    right=df['text_a'][i]
    for j in right:
        if j in m:
            right=right.replace(j,'')
        else:
            continue
    u=random.randint(0,len(right)-1)
    v=random.randint(0,len(right)-1)
    w=random.randint(0,len(right)-1)
    x=random.randint(0,len(right)-1)
    y=random.randint(0,len(right)-1)
    if len(right)<30:
        right=right.replace(right[u:u+2],right[v:v+2])
    elif len(right)<50 and len(right)>30:
        right=right.replace(right[u:u+2],right[v:v+2])
        right=right.replace(right[v:v+2],right[w:w+2])
    elif len(right)<70 and len(right)>50:
        right=right.replace(right[u:u+2],right[w:w+2])
        right=right.replace(right[w:w+2],right[v:v+2])
        right=right.replace(right[v:v+2],right[x:x+2])
    else:
        right=right.replace(right[u:u+2],right[x:x+2])
        right=right.replace(right[x:x+2],right[y:y+2])
        right=right.replace(right[y:y+2],right[v:v+2])
        right=right.replace(right[v:v+2],right[w:w+2])
    temp.append(left)
    temp.append(right)
    new2.append(temp)
df3=pd.DataFrame(new2,columns=['label','text_a'])
print(df3[:5])
print(len(df3))


# In[51]:


m='0123456789'
new3=[]
import random
for i in range(len(df)):
    temp=[]
    left=df['label'][i]
    right=df['text_a'][i]
    for j in right:
        if j in m:
            right=right.replace(j,'')
        else:
            continue
    u=random.randint(0,len(right)-1)
    v=random.randint(0,len(right)-1)
    w=random.randint(0,len(right)-1)
    x=random.randint(0,len(right)-1)
    y=random.randint(0,len(right)-1)
    if len(right)<30:
        right=right.replace(right[u:u+2],right[v:v+2])
    elif len(right)<50 and len(right)>30:
        right=right.replace(right[u:u+2],right[v:v+2])
        right=right.replace(right[v:v+2],right[w:w+2])
    elif len(right)<70 and len(right)>50:
        right=right.replace(right[u:u+2],right[w:w+2])
        right=right.replace(right[w:w+2],right[v:v+2])
        right=right.replace(right[v:v+2],right[x:x+2])
    else:
        right=right.replace(right[u:u+2],right[x:x+2])
        right=right.replace(right[x:x+2],right[y:y+2])
        right=right.replace(right[y:y+2],right[v:v+2])
        right=right.replace(right[v:v+2],right[w:w+2])
    temp.append(left)
    temp.append(right)
    new3.append(temp)
df4=pd.DataFrame(new3,columns=['label','text_a'])
print(df4[:5])
print(len(df4))


# In[52]:


# df=df.append(df1)
df=df.append(df2)
# df=df.append(df3)
df=df.append(df4)
print(len(df))


# In[48]:


print(len(df1))


# In[53]:


df.to_csv('eda_p.tsv',sep='\t',header=True,index=False)


# 下载模型

# In[33]:


get_ipython().system('git clone https://github.com/PaddlePaddle/PALM.git')


# In[35]:


get_ipython().system('cp -r PALM/paddlepalm PALM/examples/classification')


# In[1]:


get_ipython().system('cp PALM/examples/classification/run.py PALM/examples/classification/run1.py')


# In[11]:


get_ipython().system("wget 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz'")


# In[2]:


get_ipython().system("wget 'https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz'")


# In[86]:


get_ipython().system("wget 'https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz'")


# In[1]:


get_ipython().system("wget 'https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz'")


# In[20]:


get_ipython().system("wget 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz'")


# In[6]:


get_ipython().system('tar zxvf ERNIE_stable-1.0.1.tar.gz -C p/')


# In[21]:


get_ipython().system('tar zxvf chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz -C Roberta_large/')


# In[12]:


get_ipython().system('tar zxvf chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz -C Roberta_base/')


# In[2]:


get_ipython().system('tar zxvf ERNIE_1.0_max-len-512.tar.gz -C ernie_large/')


# In[87]:


get_ipython().system('tar zxvf chinese_L-12_H-768_A-12.tar.gz -C b/')


# In[8]:


import pandas as pd
data=[]
with open('Test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line=line.strip()
        tmp_sents = []
        tmp_sents.append(0)
        tmp_sents.append(line)
        data.append(tmp_sents)
df=pd.DataFrame(data,columns=['label','text_a'])
df.to_csv('pre.tsv',sep='\t',index=False,header=True)


# In[ ]:


get_ipython().system('python PALM/examples/classification/run1.py')
get_ipython().system('python PALM/examples/classification/run2.py')


# In[9]:


get_ipython().system('cp PALM/examples/classification/run1.py PALM/examples/classification/run2.py')


# 指定预测

# In[55]:


import pandas as pd
import json
import numpy as np
def res_evaluate(res_dir="./outputs7/predict/predictions.json", eval_phase='test'):
    if eval_phase == 'test':
        data_dir="pre.tsv"
    elif eval_phase == 'dev':
        data_dir="pre.tsv"
    else:
        assert eval_phase in ['dev', 'test'], 'eval_phase should be dev or test'

    preds = []
    with open(res_dir, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['label']
            preds.append(str(pred))
    df=pd.DataFrame()
    df['label']=preds
    df.to_csv('fin3.csv',index=False)

res_evaluate()
   


# In[56]:


import pandas as pd
df=pd.read_csv('fin3.csv')
print(len(df))
print(df[:5])


# In[71]:


df['label1'] = df['label'].map({0:'财经',1:'彩票',2:'房产',3:'股票',4:'家居',5:'教育',6:'科技',7:'社会',8:'时尚',9:'时政',10:'体育',11:'星座',12:'游戏',13:'娱乐'})


# In[72]:


print(len(df))


# In[73]:


print(df['label1'] [:5])


# In[74]:


df['label1'].to_csv('submission1.txt',index=False)


# In[ ]:


df['label1'].to_csv('my.csv',index=False)


# In[75]:


get_ipython().system('rm -rf submit.sh')
get_ipython().system('wget -O submit.sh http://ai-studio-static.bj.bcebos.com/script/submit.sh')
get_ipython().system('sh submit.sh submission1.txt 7abbc535bcc541c7afe59a0cee03fe7c')


# In[36]:


pre=pd.read_csv('pre.tsv',sep='\t')
print(pre[:30])


# In[39]:


s=pd.concat([pre['text_a'],df['label1']],axis=1)  #在横向合并


# In[40]:


s.to_csv('fin4.csv',index=False)


# In[74]:


w=pd.read_csv('all_data.tsv',sep='\t')
print(len(w))
print(len(w[w['label']==0]))
print(len(w[w['label']==13])/len(w))


# merge结果

# In[77]:


# df=pd.DataFrame(columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n'])
import json
import numpy as np
import pandas as pd
preds1=[]
with open('outputs1-8/outputs1/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds1.append(probes)
preds1=np.array(preds1)
preds2=[]        
with open('outputs1-8/outputs2/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds2.append(probes)
preds2=np.array(preds2)
preds3=[]      
with open('outputs1-8/outputs3/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds3.append(probes)
preds3=np.array(preds3)
preds4=[]      
with open('outputs1-8/outputs4/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds4.append(probes)
preds4=np.array(preds4)
preds5=[]      
with open('outputs1-8/outputs5/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds5.append(probes)
preds5=np.array(preds5)
preds6=[]      
with open('outputs1-8/outputs6/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds6.append(probes)
preds6=np.array(preds6)
preds7=[]      
with open('outputs1-8/outputs7/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds7.append(probes)
preds7=np.array(preds7)
preds8=[]      
with open('outputs1-8/outputs8/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds8.append(probes)
preds8=np.array(preds8)
preds9=[]      
with open('outputs9/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds9.append(probes)
preds9=np.array(preds9)
preds10=[]      
with open('outputs10/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds10.append(probes)
preds10=np.array(preds10)
preds11=[]      
with open('outputs11/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds11.append(probes)
preds11=np.array(preds11)
preds12=[]      
with open('outputs12/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds12.append(probes)
preds12=np.array(preds12)
preds13=[]      
with open('outputs13/predict/predictions.json', "r") as file:
        for line in file.readlines():
                line = json.loads(line)
                probes = line['probs']
                preds13.append(probes)
preds13=np.array(preds13)
new=1/13*preds1+1/13*preds2+1/13*preds3+1/13*preds4+1/13*preds5+1/13*preds6+1/13*preds7+1/13*preds8+1/13*preds9+1/13*preds10+1/13*preds11+1/13*preds12+1/13*preds13
# new=1/3*preds11+1/3*preds12+1/3*preds13
print(new[:5])
df=pd.DataFrame()
# df.columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
df['label']=np.argmax(new,axis=1)
df['m']=np.max(new,axis=1)


# df['all']=0.3*df['pro1']+0.4*df['pro2']+0.3*df['pro3']
# df['pred']=0
# df.loc[df['all']>=0.5,'pred']=1
# df.loc[df['all']<0.5,'pred']=0
df['result']=df['label'].astype(str)


# df.to_csv('fin5.csv',index=False)


# In[78]:


pre=pd.read_csv('pre.tsv',sep='\t')
pre['prob']=df['m']
pre['label']=df['label']
print(pre[:5])


# 建立伪标签  重复

# In[79]:


useful=pre[pre['prob']>0.9][['label','text_a']]
print(len(useful))
print(len(pre))
print(useful)


# In[80]:


train=pd.read_csv('7.tsv',sep='\t')
print(len(train))
new_train = pd.concat([train, useful]).reset_index(drop=True)
print(len(new_train))
print(new_train[:5])


# In[81]:


new_train.to_csv('pseudo_7.tsv',sep='\t',index=False,header=True)


# In[54]:


a=pd.read_csv('pseudo.tsv',sep='\t')
print(len(a))
print(a[:5])


# In[29]:


get_ipython().system('cp outputs/predict10/predictions.json outputs10/')


# In[ ]:





# In[76]:


data=pd.read_csv('all_data.tsv',sep='\t')
print(len(data))
df=data[:int(len(data)*0.7)]
print(len(df))
print(df[:5])
df.to_csv('7.tsv',sep='\t',header=True,index=False)


# In[ ]:




