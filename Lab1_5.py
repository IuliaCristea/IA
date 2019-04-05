def precision_recall_score(y_true,y_pred):
	tp=0
	fp=0
	fn=0
	for(a,b) in zip(y_true,y_pred):
		if(a==b and a==True):
			tp+=1
		if(b==True and  a==False): 
			fp+=1
		if(a==True and b==False): 
			fn+=1
	Prec=tp/(tp+fp)
	Recall=tp/(tp+fn)
	return Prec,Recall
	
y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
print(precision_recall_score(y_true,y_pred))
