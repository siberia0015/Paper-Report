from helper import *
from data_loader import *

# sys.path.append('./')
from model.models import *

class Runner(object):

	# 加载数据 （基本一致）
	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""
		print('load data...')
		ent_set, rel_set = OrderedSet(), OrderedSet()
		# 依次打开名为self.p.dataset的训练集、测试集和验证集
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				# 这里的sub, rel, obj就是数据集中的头实体、关系、尾实体
				# 如：
				# sub:/m/09c7w0
				# rel:/location/location/contains
				# obj:/m/0sqc8
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				# 将实体和关系加入实体集合和关系集合
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)
		# 字典对象{key1 : value1, key2 : value2 }
		# 根据集合生成字典对象
		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		# dict.update(dict2)把字典dict2的键/值对更新到dict里
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})
		# 与上面的字典对象key、value对调
		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
		# 实体和关系集合的元素个数
		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		# 嵌入维度（应该不用改）
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
		# defaultdict(<class 'list'>, {})
		self.data = ddict(list)
		# defaultdict(<class 'set'>, {})
		sr2o = ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				# 在self.data中添加各集合对应的三元组
				self.data[split].append((sub, rel, obj))
				# 生成训练集的sr2o
				if split == 'train':
					# 生成[(头实体，关系):尾实体]的字典
					sr2o[(sub, rel)].add(obj)
					# 生成[(尾实体，逆关系？):头实体]的字典
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		# 包含实验数据的字典
		self.data = dict(self.data)
		# (2519, 108): {90}形式
		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		# 生成测试集和验证集的sr2o
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)
		# 换种形式？
		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)
		# 遍历sr2o
		for (sub, rel), obj in self.sr2o.items():
			# 下采样？
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				# rel_inv：逆关系
				rel_inv = rel + self.p.num_rel
				# 保存三元组及其标签
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)
		# 返回一个DataLoader对象
		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}
		# 获取混洗参数
		self.chequer_perm = self.get_chequer_perm() # different
		# 获取邻接矩阵
		self.edge_index, self.edge_type = self.construct_adj()

	# 为GCN构造邻接矩阵 （完全一致）
	def construct_adj(self):
		"""
		Constructor of the runner class

		Parameters
		----------
		
		Returns
		-------
		Constructs the adjacency matrix for GCN
		# adjacency matrix：邻接矩阵
		"""
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	# 初始化Runner （完全一致）
	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		print('init Runner...')
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p))
		# pprint()模块打印出来的数据结构更加完整，每行为一个数据结构，更加方便阅读打印输出结果
		pprint(vars(self.p))
		# 是否GPU
		if self.p.gpu != '-1' and torch.cuda.is_available():
			print('gpu')
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			print('cpu')
			self.device = torch.device('cpu')
		# 加载数据
		self.load_data()
		# 设置模型
		self.model        = self.add_model(self.p.model, self.p.score_func)
		# 设置优化器
		self.optimizer    = self.add_optimizer(self.model.parameters())

	# 设置模型 （基本一致）
	def add_model(self, model, score_func):
		"""
		Creates the computational graph

		Parameters
		----------
		model_name:     Contains the model name to be created
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		print('add model...')
		model_name = '{}_{}'.format(model, score_func)
		# 根据参数选择模型
		print('model: ' + model_name.lower())
		if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.edge_index, self.edge_type,self.chequer_perm, params=self.p) # different
		else: raise NotImplementedError
		# 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
		model.to(self.device)
		# 返回模型
		return model

	# 设置优化器 （完全一致）
	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer for learning the parameters of the model
		
		"""
		print('add optimizer...')
		# Adam优化算法
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	# （完全一致）
	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split

		
		Returns
		-------
		Head, Relation, Tails, labels
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	# 获取混洗参数（新增）
	def get_chequer_perm(self):
		"""
		Function to generate the chequer permutation required for InteractE model

		Parameters
		----------

		Returns
		-------

		"""
		print('get chequer perm...')
		ent_perm = np.int32([np.random.permutation(200) for _ in range(2)])
		rel_perm = np.int32([np.random.permutation(200) for _ in range(2)])

		comb_idx = []
		for k in range(2):
			temp = []
			ent_idx, rel_idx = 0, 0

			for i in range(20):
				for j in range(10):
					if k % 2 == 0:
						if i % 2 == 0:
							temp.append(ent_perm[k, ent_idx]);
							ent_idx += 1;
							temp.append(rel_perm[k, rel_idx] + 200);
							rel_idx += 1;
						else:
							temp.append(rel_perm[k, rel_idx] + 200);
							rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]);
							ent_idx += 1;
					else:
						if i % 2 == 0:
							temp.append(rel_perm[k, rel_idx] + 200);
							rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]);
							ent_idx += 1;
						else:
							temp.append(ent_perm[k, ent_idx]);
							ent_idx += 1;
							temp.append(rel_perm[k, rel_idx] + 200);
							rel_idx += 1;

			comb_idx.append(temp)

		chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
		return chequer_perm

	# （基本一致）
	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

		Parameters
		----------
		save_path: path where the model is saved
		
		Returns
		-------
		"""
		print('save model...')
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val, # different
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	# （完全一致）
	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		print('load model...')
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	# 在验证集或测试集上评估模型（较不一致）
	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		print('在验证集或测试集上评估模型...')
		self.logger.info('在{}上评估模型...'.format(split))
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		# different（加上了hit@1和hit@10）
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5},hit@1 : {:.5}, hit@10 : {:.5}'.format(epoch,
																												 split,
																												 results[
																													 'left_mrr'],
																												 results[
																													 'right_mrr'],
																												 results[
																													 'mrr'],
																												 results[
																													 'hits@1'],
																												 results[
																													 'hits@10']))

		writeList = ['interaE',
					 '%s Set' % split, '%.6f' % results['mr'], '%.6f' % results['mrr'], '%.6f' % results['hits@1'],
					 '%.6f' % results['hits@3'], '%.6f' % results['hits@10'], '%5d' % epoch]
		os.makedirs('./logfiles/', exist_ok=True)
		with open(os.path.join('./logfiles/', self.p.dataset + self.p.model + self.p.score_func + '.txt'), 'a') as fw:
			fw.write('\t'.join(writeList) + '\n')
		# /different
		return results

	# （基本一致）
	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		# print('predict...')
		self.model.eval()

		with torch.no_grad():
			results = {}
			# 根据split和mode获取分割后的数据集
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				# pred是模型返回的得分
				pred			= self.model.forward(sub, rel) # different
				# torch.arange()创建一个一维张量，后面一个参数是设置GPU或者CPU
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				# torch.where()函数的作用是按照一定的规则合并两个tensor类型。
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				# 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。例如：x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float() # 将整数和字符串转换成浮点数
				# torch.numel返回元素数目
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					# 计算hit@n
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results

	# 所有训练样本训练一次 （完全一致）
	def run_epoch(self, epoch, val_mrr = 0):
		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		print('run epoch...')
		self.model.train()
		losses = []
		# 生成迭代器
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			# 梯度清0
			self.optimizer.zero_grad()
			# sub, rel, obj, label已经是张量了
			sub, rel, obj, label = self.read_batch(batch, 'train')

			pred	= self.model.forward(sub, rel)
			loss	= self.model.loss(pred, label)
			# 反向传播求出每个节点的梯度
			loss.backward()
			# 对模型的每个参数进行调优
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	# 模型的训练以及评估 （较不一致）
	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		----------
		
		Returns
		-------
		"""
		print('开始对模型进行训练以及评估')
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', self.p.name) # different
		print('模型保存的路径:' + save_path)
		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0 # different
		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr)
			val_results = self.evaluate('valid', epoch)
			# 如果效果好，则更新模型
			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0 # different
			# different
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25: 
					self.logger.info("Early Stopping!!")
					break
			# /different
			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))
		self.logger.info('Loading best model, Evaluating on Test data')
		pprint(vars(self.p))
		# 加载最好的模型
		self.load_model(save_path)
		test_results = self.evaluate('test', epoch)

if __name__ == '__main__':
	print('begin')
	# argparse是一个Python模块：命令行选项、参数和子命令解析器
	# 创建解析器
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# 添加参数
	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237，WN18RR') # different
	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=350,  	help='Number of epochs') # different
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate') # 学习率
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=3,   	type=int, 	help='Number of GCN Layers to use') # different
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0,  	type=float,	help='ConvE: Feature Dropout') # different
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	# 解析参数
	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
	# 参数设置
	# 设置GPU
	set_gpu(args.gpu)
	# 生成指定随机数
	np.random.seed(args.seed)
	# 设计随机初始化种子
	torch.manual_seed(args.seed)
	# 训练模型
	model = Runner(args)
	# 模型的训练以及评估
	model.fit()
	print('end')