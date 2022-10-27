import time
import OpenKE.openke as openke
from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE, TransR, TransD, TransH
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader
from bertmodel import get_bertcls_emb, get_rel_emb, load_emb

def train(transe, train_dataloader, test_dataloader = None):
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0), # define the loss function
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
	trainer.run()
	timestamp = time.strftime('%m%d%H%M', time.localtime())
	transe.save_checkpoint('./checkpoint/transe_drug_{}.ckpt'.format(timestamp))
	print('Save model: ', timestamp)

	# test
	if test_dataloader is not None:
		tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
		mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = True)
	return transe

def train_transe(mod = 'bert', model_time = '10251007', emb_dim = 768):
	train_dataloader = TrainDataLoader(
		in_path = "./drugdata/", 
		nbatches = 100,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = 25,
		neg_rel = 0)
	test_dataloader = TestDataLoader("./drugdata/", "link", True)

	if mod == 'bert':
		bert_emb, rel_emb = load_emb(model_time)
		pretrain_emb = [bert_emb, rel_emb]
		# emb_dim = 768
	else:
		pretrain_emb = None
		# emb_dim = 200
	
	model = TransE(
		ent_tot = train_dataloader.get_ent_tot(), # total number of entity
		rel_tot = train_dataloader.get_rel_tot(),
		dim = emb_dim, 
		p_norm = 1, 
		norm_flag = True,
		pretrain_emb = pretrain_emb
		)

	train(model, train_dataloader, test_dataloader)

if __name__ == '__main__':
	train_transe('bert-rel')
