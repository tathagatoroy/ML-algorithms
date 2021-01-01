import network
import  mnist

( training_data,validation_data,test_data,training_size, validation_size, test_size )= mnist.load_data_wrapper()
roy_net = network.neuralnet([784,50,10])


#print(test_data)
roy_net.SGD(training_data,30,15,1)


val = roy_net.evaluate(validation_data)
print("Result of validation : {0} pictures identified correctly out of {1} ".format(val,validation_size))
test = roy_net.evaluate(test_data)
print("Result of testing  : {0} pictures identified correctly out of {1} ".format(test,test_size))