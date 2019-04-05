def mae(y_true,y_pred):
	s=0
	for(a,b) in zip(y_true,y_pred):
		s+=abs(a-b)
	s/=len(y_true)
	return s
	
y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
print(mae(y_true,y_pred))
