from helper import *
import torch
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
from torch.nn.utils import spectral_norm

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel	,params=None):
		print('初始化模型CompGCNBase...')
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		# 初始化嵌入
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		print('初始化完成')

	def forward_base(self, sub, rel, drop1, drop2):
		# python的三目运算符
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x

class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		print('初始化模型CompGCN_TransE...')
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		print('初始化完成')

	def forward(self, sub, rel):


		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		# 实体和关系的特征向量进行拼接
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		# 最后使用sigmoid激活得到每个三元组的得分
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		print('初始化模型CompGCN_DistMult...')
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		print('初始化完成')

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

# class Self_Attn(torch.nn.Module):
# 	""" Self attention Layer"""
#
# 	def __init__(self, in_dim, activation):
# 		super(Self_Attn, self).__init__()
# 		self.chanel_in = in_dim
# 		self.activation = activation
#
# 		# self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
# 		self.query_conv =  torch.nn.Conv2d(1, out_channels=in_dim//8, kernel_size=(7, 7), stride=1, padding=0, bias=True)
# 		# self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
# 		self.key_conv = torch.nn.Conv2d(1, out_channels=in_dim//8, kernel_size=(7, 7), stride=1, padding=0, bias=True)
# 		# self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
# 		self.value_conv = torch.nn.Conv2d(1, out_channels=in_dim, kernel_size=(7, 7), stride=1, padding=0, bias=True)
# 		self.gamma = torch.nn.Parameter(torch.zeros(1))
#
# 		self.softmax =torch.nn.Softmax(dim=-1)  #
#
# 	def forward(self, x):
# 		"""
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
# 		print(x.shape)
# 		m_batchsize, C, width, height = x.size()
# 		t = self.query_conv(x)  # B X CX(N)
# 		print(t.shape)
# 		proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0,2,1)  # B X CX(N)
# 		proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
# 		energy = torch.bmm(proj_query, proj_key)  # transpose check
# 		attention = self.softmax(energy)  # BX (N) X (N)
# 		proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
#
# 		out = torch.bmm(proj_value, attention.permute(0, 2, 1))
# 		out = out.view(m_batchsize, C, width, height)
#
# 		out = self.gamma * out + x
# 		return out, attention

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
# 自注意力
class Self_Attn(torch.nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=100, kernel_size=(7,7), stride=1, padding=0)
        # self.snconv1x1_final = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,3), stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=100, kernel_size=(7,7), stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=200, kernel_size=(7,7), stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=200, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = torch.nn.Softmax(dim=-1)
        self.sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1,100, (h-6)*(w-6))
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, 100, (h-6)*(w-6)//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, 200, (h-6)*(w-6)//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, 200, h-6, w-6)
        # attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = attn_g
        return out


class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type,chequer_perm, params=None):
		print('初始化模型CompGCN_ConvE...')
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		# Batchnorm（批规范化）是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法，可以说是目前深度网络必不可少的一部分。
		self.bn0 = torch.nn.BatchNorm2d(2) # different
		# self.bn1 = torch.nn.BatchNorm2d(8)
		self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
		self.bna = torch.nn.BatchNorm2d(self.p.num_filt) # different
		self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
		# dropout：将hidden layer中的某些隐藏单元以一定的概率进行丢弃。
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		# nn.Conv2d：对由多个输入平面组成的输入信号进行二维卷积
		self.m_conv1 		= torch.nn.Conv2d(in_channels=2, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias) # different
		self.chequer_perm = chequer_perm # different
		flat_sz_h 			= int(2 * self.p.k_w) 	- self.p.ker_sz + 1
		flat_sz_w 			= self.p.k_h 			- self.p.ker_sz + 1
		self.flat_sz 		= (flat_sz_h) * (flat_sz_w ) * self.p.num_filt
		# self.flat_sz = 6 * 6* self.p.num_filt
		# nn.Linear：可以对输入数据进行线性变换
		self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		# self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        # self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		self.attn1 = Self_Attn(2) # different
		# 与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。
		self.sigma = torch.nn.Parameter(torch.zeros(1))  # different
		print('初始化完成')


	# 将e1和rel的嵌入向量拼接（较大改动）
	def concat(self, e1_embed, rel_embed):
		# e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		# rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		# 拼接e1和rel的嵌入向量
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		# 拼接后进行混洗
		chequer_perm = stack_inp[:, self.chequer_perm]
		# 混洗后将一维向量reshape成二维
		stack_inp= chequer_perm.reshape((-1, 2, 2, 200))
		# 在两个维度间进行转置
		stack_inp= torch.transpose(stack_inp, 3, 2).reshape((-1, 2, 2*self.p.k_w, self.p.k_h))
		# stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	# def concat(self, e1_embed, rel_embed):
	# 	e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
	# 	rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
	# 	stack_inp	= torch.cat([e1_embed, rel_embed], 1)
	# 	stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
	# 	return stack_inp

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp			= self.concat(sub_emb, rel_emb) # 拼接实体e1和关系re1（ConvE模型图中第一步到第二步）
		x				= self.bn0(stk_inp)		# 批规范化(x'=f(x))
		y				= x 					# 备份一份
		x               = self.m_conv1(x)		# 二维卷积（ConvE模型图中第二步到第三步）
		attention       = self.attn1(y)			# 自注意力
		# x             =self.m_conv2(x)
		x				= self.bn1(x)			# 批规范化
		attention       = self.bna(attention)	# 批规范化
		attention       = F.relu(attention)		# 激活函数
		x       		= x*attention			# 乘上自注意力权重
		x				= F.relu(x)				# 激活函数
		# x				= self.fc
		x				= self.feature_drop(x)	# 特征图dropout（ConvE模型图中第三步dropout）
		x				= x.view(-1,self.flat_sz)
		# attention		= attention.view(-1,self.flat_sz)
		# x		        = torch.cat([x,attention],1)
		x				= self.fc(x)			# 全连接层（ConvE模型图中第三步到第四步）
		x				= self.hidden_drop2(x)	# 隐层dropout（ConvE模型图第四步）
		x				= self.bn2(x)			# 批规范化
		x				= F.relu(x)		 		# 激活函数
		# 对矩阵mat1和mat2进行相乘
		x = torch.mm(x, all_ent.transpose(1,0))	# 矩阵与所有实体矩阵相乘（ConvE模型图第四步到第五步）
		x += self.bias.expand_as(x)				# 偏置向量
		score = torch.sigmoid(x)				# 激活函数（ConvE模型图第五步到第六步）
		return score

	# def forward(self, sub, rel):
    #
	# 	sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
	# 	stk_inp				= self.concat(sub_emb, rel_emb)
	# 	x				= self.bn0(stk_inp)
	# 	x				= self.m_conv1(x)
	# 	x				= self.bn1(x)
	# 	x				= F.relu(x)
	# 	x				= self.feature_drop(x)
    #
	# 	x				= x.view(-1, self.flat_sz)
    #
	# 	x				= self.fc(x)
    #
	# 	x				= self.hidden_drop2(x)
	# 	x				= self.bn2(x)
	# 	x				= F.relu(x)
    #
	# 	x = torch.mm(x, all_ent.transpose(1,0))
	# 	x += self.bias.expand_as(x)
    #
	# 	score = torch.sigmoid(x)
	#
	# 	return score
