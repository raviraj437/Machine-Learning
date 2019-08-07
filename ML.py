#importing module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing csv file of facebook
data=pd.read_csv('f:/pythonfolder/facebook.csv')
X=data.iloc[:,4:19].values;

# repacing the missing datas with their mean of column
from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN' ,strategy ='mean' ,axis=0);
im=im.fit(X[:,4:19])
X[:,4:19]=im.transform(X[:,4:19]);


#list of all column of facebook.csv
print(data.columns);

# accessing any particular block
print(data.iloc[3,6]);


from sklearn.cross_validation import train_test_split

"""#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=10)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);


#linear model
from sklearn.linear_model import LinearRegression
obj=LinearRegression()
obj.fit(x_train,y_train)
y_pred2=obj.predict(x_train);


#plot of actual point
plt.scatter(x_train,y_train,color='green')
plt.title('location of bus according to time')
plt.xlabel('time in 2400 format')
plt.ylabel('position of bus on scale 1 to 24');
plt.show()"""



# comparision of life time post consumer with indivdual factors
# lifetime postconsumer #problem set
lifetimepostconsumer = data.iloc[: ,[10]].values;
y=lifetimepostconsumer;

#page total like
pagetotallike=data.iloc[: ,[0]].values;
x=pagetotallike;

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('page total like')
plt.show()


plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('page total like')
plt.show()

"""from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,lr.predict(x_train));
"""


#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('page total like')
plt.show()
y_pred=lin.predict(x_poly);
print('here we can say that pages at like stat of around 90000 is at peak popular and most growing');


#sstotal= summation(y-ymean)^2
#ssreg=summation(ypred-ymean)^2
#ssres +ssreg =sstotal
#r^2=1-ssres/sstotal

y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r1=(1-ssres/sstotal)**(0.5);
print(r1*100);






# post month
postmonth=data.iloc[: ,[3]].values;
x=postmonth;

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('month in which page was posted')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('month in which page was posted')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=6)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('month in which page was posted')
plt.show()

print('here we can say that pages in montyh of february to march is at peak popular and most growing');



y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r2=(1-ssres/sstotal)**(0.5);
print(r2*100);



# post day
postday=data.iloc[: ,[4]].values;
x=postday;

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('day on which is page was posted')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('day on which is page was posted')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=15)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('day on which is page was posted')
plt.show()

print('here we can say that any particular day dosent affect much to ltc');



y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r3=(1-ssres/sstotal)**(0.5);
print(r3*100);





# post hour
posthour=data.iloc[: ,[5]].values;
x=posthour;

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('time at which is page was posted')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('time at which is page was posted')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('time at which is page was posted')
plt.show()

print('here we can say that between 6 am to 3 pm it is at its optimal level');



y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r4=(1-ssres/sstotal)**(0.5);
print(r4*100);





# paid
paid=data.iloc[: ,[6]].values;

x=paid;
#print(x[:,[0]]);

# repacing the missing datas with their most frequent data of column
from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN' ,strategy ='most_frequent' ,axis=0);
im=im.fit(x[:,[0]])
x[:,[0]]=im.transform(x[:,[0]]);

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('paid or unpaid')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('paid or unpaid')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('paid or unpaid')
plt.show()

print('here we can say that between paid promotion really matters');



y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r5=(1-ssres/sstotal)**(0.5);
print(r5*100);



#total reach 
totalreach=data.iloc[: ,[7]].values;
x=totalreach;
# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('reach of page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('reach of page')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('reach of page')
plt.show()

print('here we can say that maximum reach result in maximum ltc');


y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r6=(1-ssres/sstotal)**(0.5);
print(r6*100);



#total impression
repo=data.iloc[: ,[8]].values;
x=repo;
# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('reputation of page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('reputation of page')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('reputation of page')
plt.show()

print('here we can say that the users who joined the page is more important and crucial for reputstion of page');


y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r7=(1-ssres/sstotal)**(0.5);
print(r7*100);




#totalengaed users
client=data.iloc[: ,[9]].values;


#total post consumption
consumptions=data.iloc[: ,[11]].values;
x=consumptions;
# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('consumtions of post on page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('consumtions of post on page')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=10)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('consumtions of post on page')
plt.show()

print('here we can say that the average consumptions of post mainly matters if some post gain too much popularity unconditionally then good but it is RARE'); 


y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r8=(1-ssres/sstotal)**(0.5);
print(r8*100);



##
#total customer review
review=data.iloc[: ,[12]].values;
x=review;
# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('review of page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('review of page')
plt.show()



print('here we can see that very few people review the page and it doesnt affect very much to the page'); 



y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r9=(1-ssres/sstotal)**(0.5);
print(r9*100);


#post recieved
postrecieved=data.iloc[: ,[13]].values;

#engaged people
engagedpeople=data.iloc[: ,[14]].values;


#
#total comment
comment=data.iloc[: ,[15]].values;
x=comment;
# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('comment on post of page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('comment on post of page')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=5)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('comment on post of page')
plt.show()

print('comment of newly added people to the page help or boost the page a lot'); 



y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r10=(1-ssres/sstotal)**(0.5);
print(r10*100);



#
#total like
like=data.iloc[: ,[16]].values;
x=like;
print('here a strategy of mean is fitted for missing datas');
from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN' ,strategy ='mean' ,axis=0);
im=im.fit(x[:,[0]])
x[:,[0]]=im.transform(x[:,[0]]);

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('like on post of page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('like on post of page')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('like on post of page')
plt.show()

print('same as comment like play as similar role, like of newly added people to the page help or boost the page a lot'); 


y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r11=(1-ssres/sstotal)**(0.5);
print(r11*100);


#
#total share
share=data.iloc[: ,[17]].values;
x=share;
print('here a strategy of mean is fitted for missing datas');
from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN' ,strategy ='mean' ,axis=0);
im=im.fit(x[:,[0]])
x[:,[0]]=im.transform(x[:,[0]]);

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train);
y_pred=lr.predict(x_train);
plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.title('plot of original training data')
plt.ylabel('ltc');
plt.xlabel('share of page')
plt.show()

plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lr.predict(x_train),color='darkgreen');
plt.title('plot of prediction over training data')
plt.ylabel('ltc');
plt.xlabel('share of page')
plt.show()

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('share of page')
plt.show()
print('12.same as comment sharing of page also play as similar role, sharing by newly added people help to boost the page a lot'); 


y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r12=(1-ssres/sstotal)**(0.5);
print(r12*100);





#total interaction
interaction=data.iloc[: ,[18]].values;
x=share;
print('here a strategy of mean is fitted for missing datas');
from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN' ,strategy ='mean' ,axis=0);
im=im.fit(x[:,[0]])
x[:,[0]]=im.transform(x[:,[0]]);

# divsion fo training and test set dataset
x_train,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=0);

#polynomial regression
#plot according to polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=1)
x_poly=poly.fit_transform(x_train);
poly.fit(x_poly,y_train);
lin=LinearRegression();
lin.fit(x_poly,y_train);
y_pred=lin.predict(x_poly);
#plt.scatter(x_train, y_train, color='red', label='like vs ltc')
plt.plot(x_train,lin.predict(x_poly),color='darkgreen');
plt.title('plot of prediction over training data using multiple regression')
plt.ylabel('ltc');
plt.xlabel('share of page')
plt.show()
print('answering the query of page of newly added people or quick reply help in better promotion'); 


y_mean=np.mean(y_train);
sstotal=0;
i=0;
for i in range(0,400):
 sstotal+=(y_train[i]-y_mean)**2;
 
 
ssreg=0;
for i in range(0,400):
 ssreg+=(y_pred[i]-y_mean)**2;
 
 
ssres=sstotal-ssreg;
r13=(1-ssres/sstotal)**(0.5);
print(r13*100);



print('error in overall pridiction');
print(r2+r3+r4+r6);


print('conclusion')
print('1.here we can say that pages at like stat of around 90000 is at peak popular and most growing')
print('');
print('2.here we can say that pages in montyh of february to march is at peak popular and most growing')
print('');
print('3.here we can say that any particular day dosent affect much to ltc')
print('');
print('4.here we can say that between 6 am to 3 pm it is at its optimal level implies day-time is best for promotion')
print('');
print('5.here we can say that between paid promotion really matters');
print('');
print('6.here we can say that maximum reach result in maximum ltc');
print('');
print('7.here we can say that the users who joined the page earlier is more important and crucial for reputstion of page for busssiness stability and relationship');
print('');
print('8.here we can say that the average consumptions of post mainly matters if some post gain too much popularity unconditionally then good but it is RARE'); 
print('');
print('9.here we can see that very few people review the page and it doesnt affect very much to the page'); 
print('');
print('10.comment of newly added people to the page help to boost the page a lot as it increases the rach of page'); 
print('');
print('11.same as comment , like play as similar role, like of newly added people to the page help to boost the page a lot'); 
print('');
print('12.same as comment sharing of page also play as similar role, sharing by newly added people help to boost the page a lot'); 
print('');
print('answering the query of page of newly added people or quick reply help in better promotion');
