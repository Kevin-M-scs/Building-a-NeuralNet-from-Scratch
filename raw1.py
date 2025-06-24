#Goal: to create a simple 2 layer neural network for the MNIST dataset from scratch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid #used this function for calculating AUROC as np.trapz() wasnt working
import time #to calculate training time

def display_ith_teg(i,folder):
    if(folder=="train"):
        img=df_train.iloc[i-1:i,1:].to_numpy()#slice out the data record for the ith teg and convert it to numpy array
    if(folder=="test"):
        img=df_test.iloc[i-1:i,1:].to_numpy()#slice out the data record for the ith teg and convert it to numpy array
    img=img.reshape(28,28) #need to get it into order and dimensions of original image
    plt.imshow(img,cmap='gray')#display grayscale
    plt.show()

def ReLU(X):
    X[X<=0]=0
    return X

def softmax(X):
    X=X-(np.max(X,axis=0,keepdims=True))#To avoid calculation of high exponential powers
    X_exp=np.exp(X)
    X_exp_sum=np.sum(X_exp,axis=0)
    X_softmax=X_exp/X_exp_sum
    return X_softmax

def forward_prop(X_batch,y_batch,W1,b1,W2,b2,batch_size,reg_lambda,op):
    #must return net loss of current batch and 
    N=batch_size #number of training examples per batch
    X1=X_batch
    X2=(W1@X1)+b1 #hidden layer (C X N)
    X3=ReLU(X2) #(C X N)
    X4=(W2@X3)+b2 #output layer
    Y4=softmax(X4) # matrix of order (C X N)
    softmax_scores=Y4.T #order(N X C) where C=10
    if(op=="test"):#this if condition avoids unecessary loss computation during testing cycle
        return softmax_scores
    if(op=="train"):
        y_batch=y_batch.flatten() #this is very important else you will get weird output while using advanced indexing below
        correct_class_scores=softmax_scores[range(N),y_batch]
        nlog_loss=-np.log(correct_class_scores)
        batch_loss=np.sum(nlog_loss,axis=0)/N #this is (1/N) summation Li
        reg_loss=0.5*(reg_lambda)*(np.sum(W1*W1)+np.sum(W2*W2))
        net_batch_loss=batch_loss+reg_loss
        return net_batch_loss,softmax_scores,X3,X2,X1

    
def back_prop(W1,b1,W2,b2,X3,X2,X1,ss,y_batch,batch_size,learn_rate,reg_lambda):
    y_batch=y_batch.flatten()#made it 1D
    N=batch_size
    ccs=ss[range(N),y_batch]

    #calculating gradient of y4(correct class softmax) wrt scores
    #really good numpy indexing used , learnt a lot about masking
    #1st handle the derivatives of y4 wrt correct class scores ie Sy*(1-Sy)
    ss[range(N),y_batch]=ss[range(N),y_batch]*(1-ss[range(N),y_batch])
    #Now for derivatives of y4 wrt incorrect class scores ie -SySj
    #create a mask for all incorrect class scores
    mask=np.ones_like(ss,dtype=bool)
    mask[range(N),y_batch]=False
    #create an expanded correct class scores 
    temp=np.repeat(ccs,10)
    ccs_expanded=temp.reshape(-1,10)
    ss[mask]=(-1)*(ss[mask])*(ccs_expanded[mask])
    y4_grad_scores=ss.T #has order (C X N)

    #calculating gradient of E wrt y4 (ie softmax score of correct class) 
    y4_grad=(-1/(N*ccs))

    #calculating gradient of E wrt X4 (C X N)
    X4_grad=y4_grad*y4_grad_scores #hadamard product and taking advantage of broadcasting

    #calculating gradient of E wrt W2 and b2
    W2_grad=X4_grad@(X3.T) #(10 X 10)
    b2_grad=np.sum(X4_grad,axis=1).reshape(10,1)#important to reshape it as np.sum() returns a 1D array

    #calculating gradient of E wrt X3
    X3_grad=(W2.T)@(X4_grad) #(C X N)

    #calculating gradient of E wrt X2
    ReLU_X2_grad=np.where(X2<=0,0,1) # finding ReLU'(X2)
    X2_grad=X3_grad*ReLU_X2_grad #(C X N)

    #calculating gradient of E wrt W1 and b1
    W1_grad=X2_grad@(X1.T) #(10 X 784)
    b1_grad=np.sum(X2_grad,axis=1).reshape(10,1)

    #include the derivatives present in the regularization term
    dW2= W2_grad+(reg_lambda*W2) 
    db2= b2_grad
    dW1= W1_grad+(reg_lambda*W1)
    db1= b1_grad

    #updations
    W1=W1-(learn_rate*dW1)
    b1=b1-(learn_rate*db1)
    W2=W2-(learn_rate*dW2)
    b2=b2-(learn_rate*db2)

    return W1,b1,W2,b2

def calculator(y_test,y_pred,cl):
    #this function calculates tp,tn,fp,fn for a given class
    #OvR conversion(One vs Rest)
    y_cl_test=np.where(y_test==cl,1,0)
    y_cl_pred=np.where(y_pred==cl,1,0)
    tp=np.sum((y_cl_test==1) & (y_cl_pred==1))
    tn=np.sum((y_cl_test==0) & (y_cl_pred==0))
    fp=np.sum((y_cl_test==0) & (y_cl_pred==1))
    fn=np.sum((y_cl_test==1) & (y_cl_pred==0))
    return tp,tn,fp,fn

def precision(tp,fp):
    p=(tp/(tp+fp))
    return p

def recall(tp,fn):
    r=(tp/(tp+fn))
    return r

def F1(p,r):
    f1=(2*p*r)/(p+r)
    return f1

def FPR(tn,fp):
    sp=(tn/(tn+fp))
    return 1-sp #False Positive Rate

def ROC_coord(y_test,ss,t,cl):
    #This function calculates (FPR,TPR) for a given class and threshold
    y_cl_test=np.where(y_test==cl,1,0)
    pred_class_prob=(ss[:,cl:cl+1]).flatten()
    cl_label=np.where(pred_class_prob>t,1,0)
    tp=np.sum((y_cl_test==1) & (cl_label==1))
    tn=np.sum((y_cl_test==0) & (cl_label==0))
    fp=np.sum((y_cl_test==0) & (cl_label==1))
    fn=np.sum((y_cl_test==1) & (cl_label==0))
    fpr=(fp/(fp+tn))
    tpr=(tp/(tp+fn))
    return (fpr,tpr)

def ROC_plot(ltup,cl):
    #ltup is a list of tuples , each tuple is (FPR,TPR) for a particular threshold
    #ltup is the output of a list comprehension through a function which returns a tuple
    X=[ltup[0][0],ltup[1][0],ltup[2][0],ltup[3][0],ltup[4][0]]
    x=X
    x.sort() #We must ensure the x coords are in increasing order else the trapezoid() function will calculate a -ve AUROC
    Y=[ltup[0][1],ltup[1][1],ltup[2][1],ltup[3][1],ltup[4][1]] #Note here X and Y are lists of np.float64 datatype
    y=Y
    y.sort() #need to sort this too to make pairs match while calculating area
    #pyplot can easily handle most numpy datatypes
    AUROC=trapezoid(y,x)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Class {cl}")
    plt.plot(X,Y,color="red")
    #Note: To give an honest scale and size of the ROC Curve , plot it along the line y=x for ref
    X1=[0.2,0.4,0.6,0.8,1]
    Y1=[0.2,0.4,0.6,0.8,1]
    plt.plot(X1,Y1,color="black")#plotting the y=x line for reference
    plt.show()
    return AUROC


num_train=40000 #number of images in training set
num_test=5000 #number of images in test set

df_train=pd.read_csv(r"path_to_csv_train_file")
df_test=pd.read_csv(r"path_to_csv_test_file")
a=df_train.iloc[0:num_train,1:].to_numpy() #(N X 784)
a=a.astype(np.float64)
#Normalization of whole input data
mean_1=np.mean(a,axis=0) #mean is a 1D vector/array yet it can take part in broadcasting with a 2d numpy array
std_dev_1=np.std(a,axis=0) #again a 1D vector/array of 784 horizontal length
#Now its possible that some entire columns like those corresponding to the corners may be full of 0 pixels
#For eg its possible that the 1st pixel for every img in the train_set is 0, thus the 1st entire column of a may be 0 
#This will make the std_dev along that column =0 which is an issue when we divide the pixels of that column by the std_dev of
#that column as we will be divding by 0 which will return Nan (not a number). To avoid this we replace all 0 standard deviations with 1 or 1e-8
#This wont affect much as that column will still have 0's only
a-=mean_1 #zero centering input data, this subtraction works due to broadcasting
std_dev_1[std_dev_1==0]=1 #using advanced indexing of numpy we perform the above replacement
a/=std_dev_1 #normalization, again this division works due to broadcasting
X_train=a.T #this is the final normalized input (784 X N)
y_train=df_train.iloc[0:num_train,0:1].to_numpy() #(N X 1)

b=df_test.iloc[0:num_test,1:].to_numpy()
b=b.astype(np.float64)
mean_2=np.mean(b,axis=0)
std_dev_2=np.std(b,axis=0)
b-=mean_2
std_dev_2[std_dev_2==0]=1
b/=std_dev_2
X_test=b.T
y_test=df_test.iloc[0:num_test,0:1].to_numpy()

#Weight and Bias matrices initialization, Have use He(Kaiming) initialization method
#randn by defaut uses mean 0 and variance 1 so to convert that variance to (2/nin) we multiply by sqrt(2/nin)
W1=np.random.randn(10,784)*np.sqrt(2/784)
W2=np.random.randn(10,10)*np.sqrt(2/10)
b1=np.zeros((10,1))
b2=np.zeros((10,1))

train_set_size=num_train

#hyperparameters
reg_lambda=0.01
batch_size=50 
num_epochs=10
learn_rate=0.008

# start_time=time.time()
# #train
# li,lf=0,0
# for epoch in range(0,num_epochs):
#     count=0 #batch counter per epoch
#     # print(f"Epoch:{epoch+1}/{num_epochs}:")
#     for i in range(0,train_set_size,batch_size):
#         li=lf
#         X_batch=X_train[:,i:i+batch_size]
#         y_batch=y_train[i:i+batch_size]
#         #forward propagation
#         l,ss,x3,x2,x1=forward_prop(X_batch,y_batch,W1,b1,W2,b2,batch_size,reg_lambda,op="train")
#         lf=l
#         #backpropagation
#         W1,b1,W2,b2=back_prop(W1,b1,W2,b2,x3,x2,x1,ss,y_batch,batch_size,learn_rate,reg_lambda)
#         count=count+1
#     #     if(count%2==0):
#     #         print(f"Batch:{count},Loss:{l},Decrease:{li-lf}")
#     # print("----------------------------------------------------------")
# end_time=time.time()

# print(f"Total time taken for {num_epochs} epochs:{end_time-start_time:.6f} s")



#save the weight and bias matrices of the most accurate set of hyperarams to a .npz file
# np.savez("model_params.npz",W1=W1,b1=b1,W2=W2,b2=b2)
#Note:The weights saved correspond to those which gave 90.06% accuracy when model was trained on 40,000 training examples


#Load the weights and biases from .npz file
params=np.load("model_params.npz")
W1=params["W1"]
b1=params["b1"]
W2=params["W2"]
b2=params["b2"]


# #normal test
# ss=forward_prop(X_test,y_test,W1,b1,W2,b2,batch_size,reg_lambda,op="test") #(N X C)
# y_test=y_test.flatten()
# y_pred=np.argmax(ss,axis=1)
# correct,total=np.sum(y_test==y_pred),y_test.shape[0]
# acc=(correct/total)*100
# print(f"Accuracy:{acc:.2f}%")
# print(y_test,end="\n")
# print(y_pred)


#Calculating TP,FP,TN,FN per class
ss=forward_prop(X_test,y_test,W1,b1,W2,b2,batch_size,reg_lambda,op="test") #(N X C)
y_test=y_test.flatten()
y_pred=np.argmax(ss,axis=1)
#Class 0
tp_0,tn_0,fp_0,fn_0=calculator(y_test,y_pred,0)
#Class 1
tp_1,tn_1,fp_1,fn_1=calculator(y_test,y_pred,1)
#Class 2
tp_2,tn_2,fp_2,fn_2=calculator(y_test,y_pred,2)
#Class 3
tp_3,tn_3,fp_3,fn_3=calculator(y_test,y_pred,3)
#Class 4
tp_4,tn_4,fp_4,fn_4=calculator(y_test,y_pred,4)
#Class 5
tp_5,tn_5,fp_5,fn_5=calculator(y_test,y_pred,5)
#Class 6
tp_6,tn_6,fp_6,fn_6=calculator(y_test,y_pred,6)
#Class 7
tp_7,tn_7,fp_7,fn_7=calculator(y_test,y_pred,7)
#Class 8
tp_8,tn_8,fp_8,fn_8=calculator(y_test,y_pred,8)
#Class 9
tp_9,tn_9,fp_9,fn_9=calculator(y_test,y_pred,9)

#Calculating P,R,F1 per class
p0,r0=precision(tp_0,fp_0),recall(tp_0,fn_0)
f1_0=F1(p0,r0)
p1,r1=precision(tp_1,fp_1),recall(tp_1,fn_1)
f1_1=F1(p1,r1)
p2,r2=precision(tp_2,fp_2),recall(tp_2,fn_2)
f1_2=F1(p2,r2)
p3,r3=precision(tp_3,fp_3),recall(tp_3,fn_3)
f1_3=F1(p3,r3)
p4,r4=precision(tp_4,fp_4),recall(tp_4,fn_4)
f1_4=F1(p4,r4)
p5,r5=precision(tp_5,fp_5),recall(tp_5,fn_5)
f1_5=F1(p5,r5)
p6,r6=precision(tp_6,fp_6),recall(tp_6,fn_6)
f1_6=F1(p6,r6)
p7,r7=precision(tp_7,fp_7),recall(tp_7,fn_7)
f1_7=F1(p7,r7)
p8,r8=precision(tp_8,fp_8),recall(tp_8,fn_8)
f1_8=F1(p8,r8)
p9,r9=precision(tp_9,fp_9),recall(tp_9,fn_9)
f1_9=F1(p9,r9)

#Calculating p,r,f1 of net classifier using weighted aggregate
n_0,n_1,n_2,n_3,n_4=np.sum(y_test==0),np.sum(y_test==1),np.sum(y_test==2),np.sum(y_test==3),np.sum(y_test==4)
n_5,n_6,n_7,n_8,n_9=np.sum(y_test==5),np.sum(y_test==6),np.sum(y_test==7),np.sum(y_test==8),np.sum(y_test==9)
p_net=((n_0/num_test)*p0)+((n_1/num_test)*p1)+((n_2/num_test)*p2)+((n_3/num_test)*p3)+((n_4/num_test)*p4)+((n_5/num_test)*p5)+((n_6/num_test)*p6)+((n_7/num_test)*p7)+((n_8/num_test)*p8)+((n_9/num_test)*p9)
r_net=((n_0/num_test)*r0)+((n_1/num_test)*r1)+((n_2/num_test)*r2)+((n_3/num_test)*r3)+((n_4/num_test)*r4)+((n_5/num_test)*r5)+((n_6/num_test)*r6)+((n_7/num_test)*r7)+((n_8/num_test)*r8)+((n_9/num_test)*r9)
f1_net=((n_0/num_test)*f1_0)+((n_1/num_test)*f1_1)+((n_2/num_test)*f1_2)+((n_3/num_test)*f1_3)+((n_4/num_test)*f1_4)+((n_5/num_test)*f1_5)+((n_6/num_test)*f1_6)+((n_7/num_test)*f1_7)+((n_8/num_test)*f1_8)+((n_9/num_test)*f1_9)

print(f"Net Metrics\nPrecision:{p_net}, Recall:{r_net}, F1 Score:{f1_net}")

# print("ClassWise Metrics:")
# print(f"Class 0  P:{p0},R:{r0},F1:{f1_0}")
# print(f"Class 1  P:{p1},R:{r1},F1:{f1_1}")
# print(f"Class 2  P:{p2},R:{r2},F1:{f1_2}")
# print(f"Class 3  P:{p3},R:{r3},F1:{f1_3}")
# print(f"Class 4  P:{p4},R:{r4},F1:{f1_4}")
# print(f"Class 5  P:{p5},R:{r5},F1:{f1_5}")
# print(f"Class 6  P:{p6},R:{r6},F1:{f1_6}")
# print(f"Class 7  P:{p7},R:{r7},F1:{f1_7}")
# print(f"Class 8  P:{p8},R:{r8},F1:{f1_8}")
# print(f"Class 9  P:{p9},R:{r9},F1:{f1_9}")


#ROC and AUROC Classwise
cls=9 #choose class whose ROC Curve and AUROC you desire
thresholds_0=[0.15,0.45,0.5,0.7,0.9] #5 thresholds , to change number of thresholds you'll have to change code elsewhere too
AUROC_0=ROC_plot([ROC_coord(y_test,ss,x,cls) for x in thresholds_0],cls) #DOMO logic after learning about the power of list comprehension
print(AUROC_0)




