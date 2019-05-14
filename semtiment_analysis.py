#Libraries for graphical user interface
import tkinter as tk
from tkinter import *
from tkinter import ttk
from textblob import TextBlob
from tkinter.scrolledtext import *
import spacy
nlp = spacy.load('en')

#Libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
#-----------------------------------------------------------------
#libraries for Twitter access
import tweepy
import csv
csvFile = open('classified.csv', 'a')
csvWriter = csv.writer(csvFile)
#-------------------------------------------------------------------
#machine learning for classification
def getStringArrayFromNumberDataFrame(dataframe):
	list1 = []
	for s in dataframe.values:
		if len(str(s[0]))>0:
			list1.append(str(s[0]))
	return list1

def get_emotions(text,clf):
	text = text.split()
	X_new_counts = count_vect.transform(text)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	predicted = clf.predict(X_new_tfidf)
	positive_count = 0

	for x in predicted:
		if x==1:
			positive_count = positive_count+1
	return positive_count

def getEmotionFromText(text):
	positives = get_emotions(text,clf_positive)
	negatives = get_emotions(text,clf_negative)
	bad = get_emotions(text,clf_bad)
	if(bad>0):
		return("Bad tweet -> %s"%(text))
	else:
		if(positives-negatives)>0:
			csvWriter.writerow([text,1])
			return("Positive tweet ->%s"%(text))
		elif(negatives-positives)>0:
			csvWriter.writerow([text,-1])
			return("Negative tweet ->%s"%(text))
		else:
			csvWriter.writerow([text,0])
			return("Neutral tweet ->%s"%(text))


train_data_csv_name = "TrainingData.csv"
df_x_words = pd.read_csv(train_data_csv_name,usecols=[0],header=None)
df_y_positive = pd.read_csv(train_data_csv_name,usecols=[1],header=None)
df_y_negative = pd.read_csv(train_data_csv_name,usecols=[2],header=None)
df_y_bad = pd.read_csv(train_data_csv_name,usecols=[3],header=None)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(getStringArrayFromNumberDataFrame(df_x_words))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf_positive = MultinomialNB().fit(X_train_tfidf,df_y_positive)
clf_negative = MultinomialNB().fit(X_train_tfidf,df_y_negative)
clf_bad = MultinomialNB().fit(X_train_tfidf,df_y_bad)


def upload_file():
	outputfile = open('sentimentanalysis.txt','w')
	tweet_file_name=tk.filedialog.askopenfilename(filetypes=(("Test Files",".csv"),("All files","*")))
	res_op = "Testing datafile successfully uploaded\n"
	displayed_file.insert(tk.END,res_op)
	with open(tweet_file_name,encoding="utf-8",errors='ignore') as f:
		for line in f:
			try:
				tmpstr = getEmotionFromText(line)
				print(tmpstr)
				displayed_file.insert(tk.END,tmpstr)
				outputfile.write(tmpstr)
			except:
				pass
	outputfile.close()

#--------------------------------------------------------------------
#Acess Twitter using tweepy
def access_twitter():
	auth = tweepy.auth.OAuthHandler('gQHHduyNMWgVmfwpwSiJ4b3ck', 'Mk2dIXoNqTMTc7ZwwM0NPbY2WaU9x4JTDjZygIgIxIQjifphpd')
	auth.set_access_token('996018061890076672-vrRltZ2bSHNOMbMCSrFNGPVOBQKynzs', 'ButEbyLQTDyQoCAQnePtepWfH5qOQNJS6Scl9cm2Z6wqB')

	api = tweepy.API(auth)
	# Open/create a file to append data to
	csvFile = open('download.csv', 'a')

	csvWriter = csv.writer(csvFile)
	for tweet in tweepy.Cursor(api.search,q = "religion",since = "2019-05-08",until = "2019-05-09",lang = "en").items():
		csvWriter.writerow([tweet.text.encode('utf-8')])
		print(tweet.text)
	csvFile.close()

def view_tweet_file():
	file1= tk.filedialog.askopenfilename(filetypes=(("Test Files",".csv"),("All files","*")))
	read_text = open(file1).read()
	displayed_file.insert(tk.END,read_text)


def Metric_classifier():
	data = pd.read_csv("classified.csv",header=None)
	print("\n")
	print(data.head())
	print(data.shape)
	#----------------------------------------------------------------------------------
	# Machine Learning: Support vector Machine and Multinomial Naive Bayes
	X = data.iloc[:,0] #Accessing the column 1
	y = data.iloc[:,1] #Accessing the column 2
	print(X.shape)
	print("***************************")
	print(y.shape)
	#-------------------------------------------------------------------------------
	#Machine learning Algorithm
	#splitting the dataset into  75% training data and 25% testing data
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) 
	'''print("The dimesnions of the training set and the testing set")
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(y_test.shape)
	'''
	#---------Vectorizing our dataset-----------------------------------------------
	vect = CountVectorizer()
	vect.fit(X_train)
	X_train_dtm = vect.transform(X_train)
	X_train_dtm = vect.fit_transform(X_train)
	X_test_dtm = vect.transform(X_test)
	#--------------------------------------------------------------------------------
	nb = MultinomialNB()
	nb.fit(X_train_dtm, y_train)
	y_pred_class = nb.predict(X_test_dtm)
	acc_nb = metrics.accuracy_score(y_test, y_pred_class)
	confusion_matrix = metrics.confusion_matrix(y_test, y_pred_class)
	print("Accuracy of the NB model is: ",acc_nb*100,"%")
	acc_recall = metrics.recall_score(y_test, y_pred_class,average=None)
	acc_prec = metrics.precision_score(y_test,y_pred_class,average=None)
	text1 = "The metric of Multinomial Naive Bayes classifier: "
	text2 = "**************************************************"
	print("confusion matrix of the model is: ")
	print(confusion_matrix)
	result1 = '\n {} \n\nAccuracy: {} % \n\nPrecision: {} \n\nRecall:{} \n\nConfusion Matrix:{},\n {}'.format(text1,acc_nb*100,acc_prec,acc_recall,confusion_matrix,text2)
	tab4_display.insert(tk.END,result1)

	clf = svm.SVC(kernel='linear')
	clf.fit(X_train_dtm, y_train)
	y_pred = clf.predict(X_test_dtm)
	acc_svm = metrics.accuracy_score(y_test, y_pred)
	svm_recall = metrics.recall_score(y_test, y_pred,average=None)
	svm_prec = metrics.precision_score(y_test,y_pred,average=None)
	confusion_matrix1 = metrics.confusion_matrix(y_test, y_pred)
	text3 = "The metric of the SVM classifier: "
	print("Accuracy of the SVM model is: ",acc_svm*100,"%")
	result4 = '\n {} \n\nAccuracy: {}% \n\nPrecision: {} \n\nRecall:{} \n\n Confusion Matrix: {}'.format(text3,acc_svm*100,svm_prec,svm_recall,confusion_matrix1)
	tab4_display.insert(tk.END,result4)

#--------------------------------------------------------------------------

window = Tk()
window.title("Sentiment analysis Graphical user interface")
window.geometry("800x400")

# Tab layout
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)

#Add tabs notebook
tab_control.add(tab1,text="Sentiment analysis")
tab_control.add(tab2,text="File processor")
tab_control.add(tab4,text="Metrics")
tab_control.add(tab3,text="About")

tab_control.pack(expand=1,fill='both')

#Tab for sentiment analysis
label1=Label(tab1,text="Sentiment analysis using machine learning",padx=5,pady=5)
label1.grid(column=0,row=0)
label2=Label(tab2,text="File processing",padx=5,pady=5)
label2.grid(column=0,row=0)
label3=Label(tab3,text="About",padx=5,pady=5)
label3.grid(column=0,row=0)

def count_vectorizer():
	cv = CountVectorizer()
	raw_text = str(raw_entry.get())
	my_input=raw_text.split(",")
	word1 =cv.fit_transform(my_input)
	final_array = word1.toarray()
	final_text = cv.get_feature_names()
	result = '\n\nArray: {} \n\nFeatures:{} '.format(final_array,final_text)
	tab1_display.insert(tk.END,result)


def tfidf_trans():
	cv = TfidfVectorizer(min_df=1,stop_words='english')
	raw_text = str(raw_entry.get())
	my_input =raw_text.split(",")
	word2 = cv.fit_transform(my_input)
	final_array = word2.toarray()
	final_text = cv.get_feature_names()
	result = '\n \nArray: {} \n \nFeatures:{}'.format(final_array,final_text)
	tab1_display.insert(tk.END,result)

def get_tokens():
	raw_text = str(raw_entry.get())
	new_text = TextBlob(raw_text)
	final_text = new_text.words
	result = '\nTokens: {}'.format(final_text)
	tab1_display.insert(tk.END,result)

def get_pos_tags():
	raw_text = str(raw_entry.get())
	new_text = TextBlob(raw_text)
	final_text = new_text.tags
	result = '\nPOS of speech: {}'.format(final_text)
	tab1_display.insert(tk.END,result)

def clear_entry_text():
	entry1.delete(0,END)

def clear_display_result():
	tab1_display.delete('1.0',END)


# Display screen for result
tab1_display = ScrolledText(tab1, height=7)
tab1_display.grid(row=7,column=0,columnspan=3,padx=5,pady=5)

l1 = Label(tab1,text="Enter the text to analysis")
l1.grid(row=1,column=0)

raw_entry = StringVar()
entry1 = Entry(tab1,textvariable=raw_entry,width=50)
entry1.grid(row=1,column=1)

#Buttons on the Tkinter screen
button0 = Button(tab1,text='Tokens',width=12,bg='#03A9F4',fg='#FFF',command=get_tokens)
button0.grid(row=4,column=0,padx=10,pady=10)

button5 = Button(tab1,text='POS tags',width=12,bg='#03A9F4',fg='#FFF',command=get_pos_tags)
button5.grid(row=4,column=1,padx=10,pady=10)

button1 = Button(tab1,text='Countvectorizer',width=12,bg='#03A9F4',fg='#FFF',command=count_vectorizer)
button1.grid(row=4,column=2,padx=10,pady=10)

button2 = Button(tab1,text='Tfidf',width=12,bg='#03A9F4',fg='#FFF',command=tfidf_trans)
button2.grid(row=5,column=0,padx=10,pady=10)

button3 = Button(tab1,text='reset',width=12,bg='#03A9F4',fg='#FFF',command=clear_entry_text)
button3.grid(row=5,column=1,padx=10,pady=10)

button4 = Button(tab1,text='Clear result',width=12,bg='#03A9F4',fg='#FFF',command=clear_display_result )
button4.grid(row=5,column=2,padx=10,pady=10)
#**********************************************************************************************************************
#**********************************************************************************************************************
# Tab 2: File processing
def openfiles():
	file1= tk.filedialog.askopenfilename(filetypes=(("Test Files",".csv"),("All files","*")))
	read_text = open(file1).read()
	displayed_file.insert(tk.END,read_text)

def clear_text_file():
	displayed_file.delete('1.0',END)

def clear_result():
	tab2_display_file.delete('1.0',END)

def open_output_file():
	file1= tk.filedialog.askopenfilename(filetypes=(("Test Files",".txt"),("All files","*")))
	read_text = open(file1).read()
	displayed_file.insert(tk.END,read_text)


l1 = Label(tab2,text='open file to processor',padx=5,pady=5)
l1.grid(column=1,row=1)

displayed_file = ScrolledText(tab2,height=10)
displayed_file.grid(row=2,column=0,columnspan=3,padx=5,pady=3)

b0 = Button(tab2,text='upload testing file',width=12,bg='#b9f6ca',command=upload_file)
b0.grid(row=3,column=0,padx=10,pady=10)

b2 = Button(tab2,text='Retrieve tweets',width=12,bg='#b9f6ca',command=access_twitter)
b2.grid(row=3,column=1,padx=10,pady=10)

b3 = Button(tab2,text='Output file',width=12,bg='#b9f6ca',command=open_output_file)
b3.grid(row=3,column=2,padx=10,pady=10)

b6 = Button(tab2,text='View tweet file',width=12,bg='#b9f6ca',command=view_tweet_file)
b6.grid(row=4,column=0,padx=10,pady=10)

b4 = Button(tab2,text='Reset',width=12,bg='#b9f6ca',command=clear_text_file)
b4.grid(row=4,column=1,padx=10,pady=10)

#About tab
about_label = Label(tab3,text="Sentiment analsysis GUI V.0.0.1 \n Created using python Tkinter \n Name:Prudvi\nRoll.No: 2016BCS0011\nIndian Institute of Information Technology Kottayam",pady=5,padx=5)
about_label.grid(column=0,row=1)
#-------------------------------------------------------------------------------------------------------
#Tab4
tab4_display = ScrolledText(tab4, height=20)
tab4_display.grid(row=7,column=0,columnspan=3,padx=5,pady=5)

l1 = Label(tab4,text="Metric of the Mutlinomial Naive Byaes and SVM classifier")
l1.grid(row=1,column=0)

b1 = Button(tab4,text="Metrics",width=12,bg='#b9f6ca',command=Metric_classifier)
b1.grid(row=3,column=0,padx=10,pady=10)

window.mainloop()
