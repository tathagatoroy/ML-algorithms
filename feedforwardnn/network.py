import numpy as np 
import random
#Assumption : the network has one hidden layer other than the input output layer
#Todo : Generalise the code implementation to accomodate a more generic architecture

class neuralnet():
    #weights and architecture initialisation
    def __init__(self,dim):
        self.num_layers = len(dim)  # 3 for the purposes of this implementation
        self.dim = dim  
         #elements in the weight matrix is initialised from a normal distribution ~ N(0,1)
        self.weight = [np.random.normal(0,1,[y , x]) for x,y in zip(dim[:-1],dim[1:])]
        print("size of weight : {}".format(len(self.weight)))
        self.bias = [np.random.normal(0,1,[y , 1]) for y in dim[1:]] 
        print("size of bias : {}".format(len(self.bias)))
        
    
    #sigmoid function
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def del_sigmoid(self,z):
        return ((np.exp(-z))/((1.0+np.exp(-z))*(1.0+np.exp(-z))))

    
    #feedforward function
    def predict(self,input):
        hidden_layer_input = self.sigmoid(np.matmul(self.weight[0],input))
        output = self.sigmoid(np.matmul(self.weight[1],hidden_layer_input))
        return output
    

    #backprop for a given input x and label y ,return (grad_weight)_x and (grad_bias)_x

    def backprop(self,x,y):
        #this is useful for later appending
        #feedforward
        grad_w = [ np.zeros(b.shape) for b in self.weight] #stores the result grad_weight
        grad_b = [ np.zeros(w.shape) for w in self.bias]#stores the result grad_bias
        activation = [x] #stores the activation vector of all layers
        a = x    #useful temp var to iteratw
        z = []   #stores z vector for all layers
        #forward pass
        
        for w,b in zip(self.weight,self.bias):
            z1 = np.dot(w,a)+b
            z.append(z1)
            a = self.sigmoid(z1)
            activation.append(a)
        initial_error = (a - y)
        d_val = np.multiply(initial_error,self.del_sigmoid(z[-1]))
        
        del_val = [d_val] #stores the del values for all neurons
        grad_b[-1] = (d_val)
        grad_w[-1] = (np.dot(d_val,activation[-2].transpose())) #using negative indices
        
        for i in range(2, self.num_layers):
            d_val = np.multiply(np.dot(self.weight[-i+1].transpose(),d_val),self.del_sigmoid(z[-i]))
            del_val.append(d_val)
            # print(-i)
            #print(self.num_layers)
            #print(len(grad_b))
            grad_b[-i] = d_val
            grad_w[-i] = np.dot(d_val,activation[-i-1].transpose())

        return (grad_b,grad_w)

    #return the total number of test data it is correct and total number of test data
    def evaluate(self,test_data):
        test_result = [(np.argmax(self.predict(x)),y) for (x,y) in test_date] #finds the indices with max value in predict(x)
        return  sum(int(x == y) for (x,y) in test_result)   #counts where result is correct




        #SGD = stochastic gradient descent
        #inputs training data,number of epochs,mini_batch_size and learning rate and updates the biases/weights
        #this is the optimisation part ,learning part of the code,
        #test data to print progress
    
    #update bias part and the gradient descent
    def learn(self,mini_batch,learning_rate):
        #initialise the gradients with zero
        grad_w = [np.zeros(w.shape) for w in self.weight]
        grad_b = [np.zeros(b.shape) for b in self.bias]
        for x,y in mini_batch:
            b,w  = self.backprop(x,y) #gradients for the single training example
            grad_w = [cur + val for cur,val in zip(grad_w,w)]  #updating the value to get gradient for
            grad_b = [cur + val for cur,val in zip(grad_b,b)]  #the whole batch
        #updates weights and bias    
        self.weight = [cur_weight - (learning_rate/len(mini_batch))*gradient for cur_weight,gradient in zip(self.weight,grad_w)]
        self.bias = [cur_bias - (learning_rate/len(mini_batch))*gradient for cur_bias,gradient in zip(self.bias,grad_b)]

    
    #Computes the cost function for a data set
    def computecost(self,data):
        cost = 0.0
        for x,y in data:
            output = self.predict(x)
            diff = (output - y)*(output - y)
            cost1 = np.sum(diff)
            cost += cost1
        cost = cost/len(data)
        return cost


    def SGD(self,train_data,epochs,batch_size,learning_rate):
        train_data = list(train_data)
        size_data = len(list(train_data))
        for i in range(epochs):
            random.shuffle(list(train_data)) #returns a permutation of data
            #generates mini_batches by segmenting the shuffled training data
            mini_batches = [train_data[k : k+batch_size] for k in range(0,size_data,batch_size)] 
            for mini_batch in mini_batches :
                self.learn(mini_batch,learning_rate)
            training_cost = self.computecost(train_data)
            print("epoch {0} is finished running ,training cost : {1}".format(i,training_cost))
    







            
        

        
    
        