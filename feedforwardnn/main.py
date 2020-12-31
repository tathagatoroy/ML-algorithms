import network
import  mnist

training_data,validation_data,test_data = mnist.load_data_wrapper()
roy_net = network.neuralnet([784,50,10])
roy_net.SGD(training_data,30,15,1)
print("Validation")
roy_net.evaluate(validation_data)
print("Testing")
roy_net.evaluate(test_data)