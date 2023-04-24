dataset = 'kuai'
assert dataset in ['kuai','coat','yahoo']

# model name
model = 'GMF'
assert model in ['GMF']

# paths
main_path = '../../data/kuai/kuai-FM/'
train_rating = main_path + 'sets/training_list.npy'
test_rating = main_path + 'sets/testing_list.npy'
test_negative = main_path + '{}.testing.negative'.format(dataset)

model_path = './models/'
MF_model_path = model_path + 'GMF.pth'
