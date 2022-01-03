from docx import Document
import os
import sys
import importlib
import pickle 
importlib.reload(sys)
def Getfile(path):
    #path = path.encode('utf-8')
    File = []
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path,parent)
        #child2 = os.path.join(path2,parent)
        if os.path.isdir(child):
            File = File + Getfile(child)
        else:
            File.append(child)
    return File
def SPLIT(text):
    l = len(text)
    ans = []
    strr = ""
    cnt = 0
    for i in range(l):
        strr+= text[i]
        #if text[i] == '\n':
        #    if strr != "":
        #        ll = len(strr)
        #        ok = 0
        #        for ss in range(0,ll):
        #            if(strr[ss]!=" " and strr[ss]!="\t" and strr[ss]!="\n"):
        #                ok = 1
        #        if(ok==1):
        #            ans.append(strr)
        #    strr = ""
        #    cnt = 0
        if text[i] == '.' and cnt == 0:
            if strr != "":
                ll = len(strr)
                ok = 0
                for ss in range(0,ll):
                    if(strr[ss]!=" " and strr[ss]!="\t" and strr[ss]!="\n"):
                        ok = 1
                if(ok==1):
                    ans.append(strr)
            strr = ""
        if text[i] == '(':
            cnt += 1
        if text[i] == ')':
            if(cnt-1>=0):
                cnt -= 1
    if strr != "":
        ll = len(strr)
        ok = 0
        for ss in range(0,ll):
            if(strr[ss]!=" " and strr[ss]!="\t" and strr[ss]!="\n"):
                ok = 1
        if(ok==1):
            ans.append(strr)
    return ans

def Word(filename):
    try:
        #print(os.path.abspath(filename))
        pt = os.path.abspath(filename)
        #W = filename.decode("unicode_escape")	
#print(chardet.detect ( filename ))
        document = Document(pt)
        headings = []
        texts = []
        All =[]
        for paragraph in document.paragraphs:
            if paragraph.style.name.startswith('Heading'):
                headings.append(paragraph.text)
            if paragraph.style.name.startswith('Normal'):
                texts.append(paragraph.text)
            All.append(paragraph.text)
	#All = headings + texts
        str_texts1="".join(All)
        str_texts2 = SPLIT(str_texts1)
        #str_texts2 = []
        #for t in texts:
        #    a = SPLIT(t)
        #    for aa in a:
        #        str_texts2.append( aa )
        #str_texts2=".\n".join(str_texts1.split("."))\
        
     #   print(str_texts)
        return str_texts2
    except Exception as e:
        print(e)
        return " "
filePath = "./RGZN/Arxiv6K_docx/"
docxfiles = Getfile(filePath)
#docxfiles = ["Arxiv6K_docx/_comments.docx"]
#if (os.path.exists("Index/DS.txt") and os.path.isfile("Index/DS.txt") and os.path.getsize("Index/DS.txt") > 0):
#    with open("Index/DS.txt", "rb") as DS_Data:
#         DD = pickle.load(DS_Data)
#with open("Index/DS.txt", "wb") as DS_Data:
#        pickle.dump(DD,DS_Data)
print(len(docxfiles))
with open("Docdata.txt", "wb") as TAX:
    ans = []
    for Fpath in docxfiles:
        strr = Word(Fpath)
        if strr == []:
            continue
        ans.append(strr)
        #print(strr)
    pickle.dump(ans,TAX)
    st = 0
    for i in ans:
        st += len(i)
    print("所有句子数量:")
    print(st)
    
    
