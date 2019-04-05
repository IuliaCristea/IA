def accuracy_score(y_true,y_pred):
	s=0
	for i in range(len(y_true)):
		if(y_true[i]==y_pred[i]):
			s=s+1 
	s=s/len(y_true)
	return s
	
y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
print(accuracy_score(y_true,y_pred))

# for(