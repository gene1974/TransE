import time
import OpenKE.openke as openke
from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader


if __name__ == '__main__':
	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = "./drugdata/", 
		nbatches = 100,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = 25,
		neg_rel = 0)

	# dataloader for test
	test_dataloader = TestDataLoader("./drugdata/", "link", True)

	# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(), # total number of entity
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 200, 
		p_norm = 1, 
		norm_flag = True)

	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
	trainer.run()
	timestamp = time.strftime('%m%d%H%M', time.localtime())
	transe.save_checkpoint('./checkpoint/transe_drug_{}.ckpt'.format(timestamp))

	# test
	tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
	mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = True)
	print()

