import random
import pickle
import torch
import numpy as np
from os.path import join

m = 512

w = 32
comLower = 1

class MyNumbers:
    def __iter__(self):
        self.magni = comLower
        return self
    def __next__(self):
        self.cardi = int(pow(10, self.magni))
        if self.magni <= 9.0:
            self.magni += 1
        else:
            raise StopIteration
        
        return self.cardi

if __name__ == "__main__":
	print(f'm={m}')
	'''train'''
	data_train = np.load(f'./data/train/{m}.npy')
	data = np.array(data_train)	# (219900, 34)

	data_in = data[:, 0 : 34 - 1 - 1]
	data_out = data[:, 34 - 1]

	train_set = []
	train_set_target = []

	for i in range(data_out.size):
		train_set.append(list(data_in[i]))
		train_set_target.append([data_out[i]])

	train_set = torch.tensor(train_set, dtype = torch.float32)
	train_set_target = torch.tensor(train_set_target, dtype = torch.float32)


	print(train_set.shape)
	print(train_set_target.shape)

	fw = open('train.pkl', 'wb')
	pickle.dump(train_set, fw)
	pickle.dump(train_set_target, fw)
	fw.close()

	'''test'''
	myCardi = MyNumbers()
	array = np.array([np.load(f"./data/test/{m}-{1}0.npy")[0]])
	for i in iter(myCardi):
		eleNum = i
		magni = int(np.log10(eleNum))
		array = np.vstack((array, np.load(f"./data/test/{m}-{magni}0.npy")))

	print(np.shape(data), np.shape(array))
	data_in = array[:, 0 : 34 - 1 - 1]
	data_out = array[:, 34 - 1]

	test_set = []
	test_set_target = []

	for i in range(data_out.size):
		test_set.append(list(data_in[i]))
		test_set_target.append([data_out[i]])

	test_set = torch.tensor(test_set, dtype = torch.float32)
	test_set_target = torch.tensor(test_set_target, dtype = torch.float32)


	print(test_set.shape)
	print(test_set_target.shape)

	fw = open('test.pkl', 'wb')
	pickle.dump(test_set, fw)
	pickle.dump(test_set_target, fw)
	fw.close()