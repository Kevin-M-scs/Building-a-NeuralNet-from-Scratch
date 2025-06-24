import matplotlib.pyplot as plt

#---------------------------------------------
#Learning curve for MNIST raw1.py 
#Test size fixed at 5000
#Number of epochs=10
X=[10,15,20,30,40,50,60]#order of 1000's
Y=[85.6,87.32,88.24,88.8,90.4,89.62,90.28]

plt.xlabel("Train Size (scale of 1000)")
plt.ylabel("Test Accuracy")
plt.title("Learning Curve for Raw1.py model")
plt.plot(X,Y,color="red")
plt.show()
#----------------------------------------------


