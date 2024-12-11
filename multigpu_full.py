import torch
import torch.distributed
import torch.multiprocessing as mp
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import argparse

graph_name = 'reddit'

class Buffer(object):
    def __init__(self):
        super(Buffer, self).__init__()
        self._n_layers = 0
        self._boundary = []
        self._pl, self._pr = [], []
        self._f_buf = []
        self._f_recv, self._b_recv = [], []
        self._device = None

    #配置通信模块
    def init_buffer(self, device, num_all, boundary, f_recv_shape, layer_size):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._device = device
        self._n_layers = len(layer_size)

        self._boundary = boundary
        tot = f_recv_shape[rank]
        f_recv_shape[rank] = None
        #计算pl和pr数组
        for s in f_recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

        #self._f_buf[L]为第L层通信完毕之后的节点嵌入。
        self._f_buf = [None] * self._n_layers
        
        #self._f_recv[L]、self._b_recv[L]分别为接收第L层前向传播和反向传播数据的缓存。
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers

        for i in range(self._n_layers):

            #第L层通信完毕之后的节点嵌入张量大小应为图中所有节点个数num_all乘以模型第L层大小。
            self._f_buf[i] = torch.zeros([num_all, layer_size[i]], device=self._device)
            tmp1, tmp2 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(torch.zeros(1, device=self._device))
                    tmp2.append(torch.zeros(1, device=self._device))
                else:

                    #接收第L层前向传播的缓存张量大小。
                    s1 = torch.Size([f_recv_shape[j], layer_size[i]])

                    #接收第L层反向传播的缓存张量大小。
                    s2 = torch.Size([boundary[j].shape[0], layer_size[i]])
                    tmp1.append(torch.zeros(s1, device=self._device))
                    tmp2.append(torch.zeros(s2, device=self._device))

            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2
                                       
    def update(self, layer, feat):
        #节点嵌入的通信，通信结果保存在self._f_recv[layer]缓存中。
        self.__feat_transfer(layer, feat)
        #因为属于相同分区的HALO节点编号相邻，可以将通信结果直接拼接起来。
        self._f_buf[layer] = self.__feat_concat(layer, feat)
        #对于需要计算梯度的feature，使用register_hook()在计算本地的梯度之后更新正确的梯度。
        if self._f_buf[layer].requires_grad:
            self._f_buf[layer].register_hook(self.__grad_hook(layer))
        #返回包含正确HALO节点嵌入的feature。
        return self._f_buf[layer]
    
    def __feat_transfer(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = []
        for i in range(size):
            if i != rank:
                tmp.append(feat[self._boundary[i]])
            else:
                tmp.append(torch.zeros(1, device=self._device))
        dist.all_to_all(self._f_recv[layer], tmp)

    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                tmp.append(self._f_recv[layer][i])
        return torch.cat(tmp)
    
    def __grad_hook(self, layer):
        def fn(grad):
            #梯度通信
            self.__grad_transfer(layer, grad)
            #梯度更新
            self.__update_grad(layer, grad)
            return grad
        return fn

    def __grad_transfer(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = []
        for i in range(size):
            if i != rank:
                tmp.append(grad[self._pl[i]:self._pr[i]])
            else:
                tmp.append(torch.zeros(1, device=self._device))
        dist.all_to_all(self._b_recv[layer], tmp)
    
    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                grad[self._boundary[i]] += self._b_recv[layer][i]

class GraphSAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, buffer):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.SAGEConv(in_size, hid_size, aggregator_type='mean')
        )
        self.layers.append(
            dglnn.SAGEConv(hid_size, out_size, aggregator_type='mean')
        )
        
        #引入自定义的通信管理模块buffer用于实现节点嵌入和梯度的通信
        self.buffer = buffer

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            #前向传播中的嵌入通信
            h = self.buffer.update(i, h)
            h = layer(g, h)
        return h

def load_partition(rank):
    graph_dir = 'partitions/' + graph_name + '/'
    part_config = graph_dir + graph_name + '.json'

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]

    #提取节点数据(*@\label{line:move_node_data_l}@*)
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()

    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
    node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/train_mask')
    node_feat.pop(node_type + '/val_mask')
    node_feat.pop(node_type + '/test_mask')
    
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb

def get_boundary(device, node_dict, gpb):
    #边界节点的数量未知，首先通过通信获取当前分区p中分区q需要的节点嵌入编号数量。
    rank, size = dist.get_rank(), dist.get_world_size()
    send_num = [(node_dict['part_id'] == i).sum().view(-1) for i in range(size)]
    recv_num = [torch.tensor([0], device=device) for i in range(size)]
    boundary = [None] * size
    dist.all_to_all(recv_num, send_num)

    #通过边界节点的数量预先分配好通信所需的临时空间，再通信具体的边界节点编号。
    send_boundary = [node_dict[dgl.NID][(node_dict['part_id'] == i)] - gpb.partid2nids(i)[0].item() for i in range(size)]
    boundary = [torch.zeros(recv_num[i].item(), dtype=torch.long, device=device) for i in range(size)]
    dist.all_to_all(boundary, send_boundary)

    #将边界节点编号排序，有利于后续通信节点嵌入时的效率。
    for i in range(size):
        if i == rank: boundary[i] = None
        else: boundary[i], _ = torch.sort(boundary[i])

    return boundary

#将图重新标号
def reorder(device, graph, node_dict, boundary):
    rank, size = dist.get_rank(), dist.get_world_size()
    num_tot = graph.num_nodes()
    new_id = torch.zeros(num_tot, dtype=torch.int, device=device)

    #生成一个节点编号排列node_id表示图重标号的节点编号顺序。
    offset = 0
    node_id = (node_dict['part_id'] == rank)
    num_node = node_id.bool().sum().item()
    new_id[node_id] = torch.arange(num_node, dtype=torch.int, device=device) + offset
    offset = offset + num_node
    for i in range(size):
        if i == rank: continue
        node_id = (node_dict['part_id'] == i)
        num_node = node_id.bool().sum().item()
        new_id[node_id] = torch.arange(num_node, dtype=torch.int, device=device) + offset
        offset = offset + num_node

    u, v = graph.edges()
    u = new_id[u.long()]
    v = new_id[v.long()]
    graph = dgl.graph((u.long(), v.long()))

    #图重新标号之后，boundary的编号也需要随之改变。
    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary

def evaluate(g, features, labels, mask, model, device):
    model.eval()                                         #切换为评估模式
    with torch.no_grad():                                #评估过程无需计算梯度
        logits = model(g, features)                      #模型推理
        logits = logits[mask]
        labels = labels[mask]                            #提取出验证集的结果
        _, indices = torch.max(logits, dim=1)            #根据结果给出最终预测的分类
        correct = torch.sum(indices == labels)
        num_items = torch.tensor(len(labels), device=device)
        dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)       #通过reduce获取各个分区预测正确的节点个数
        dist.reduce(num_items, dst=0,  op=dist.ReduceOp.SUM)    #通过reduce获取各个分区验证集的节点个数
        if dist.get_rank() == 0: return correct.item() * 1.0 / num_items        #只需在rank 0计算预测精度
        else: return None

def run(proc_id, devices, n_feat, n_hidden, n_class, n_layers):
    #初始化PyTorch分布式(*@\label{line:init_pytorch_comm_l}@*)
    dev_id = devices[proc_id]
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    torch.cuda.set_device(dev_id)
    device = torch.device("cuda:" + str(dev_id))
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=len(devices),
        rank=proc_id,
    )
    rank, size = dist.get_rank(), dist.get_world_size()
    
    #载入进程对应的分区数据
    graph, node_dict, gpb = load_partition(rank)
    graph = graph.int().to(device)
    for key in node_dict.keys():
        node_dict[key] = node_dict[key].to(device)
    

    #数据预处理、通信模块配置以及训练模型的主循环代码将在后面的内容中介绍
    boundary = get_boundary(device, node_dict, gpb)
    graph, node_dict, boundary = reorder(device, graph, node_dict, boundary)
    recv_shape = [(node_dict['part_id'] == i).int().sum().item() for i in range(size)]
    
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)

    #实例化通信模块
    buffer = Buffer()
    buffer.init_buffer(device, graph.num_nodes(), boundary, recv_shape, layer_size)
    del boundary
    
    labels = node_dict['label']
    train_mask = node_dict['train_mask']
    val_mask = node_dict['val_mask']
    feat = node_dict['feat']

    #定义模型并用DDP进行包装
    model = GraphSAGE(n_feat, n_hidden, n_class, buffer)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)

    #选择合适的loss函数以及优化器
    loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    #将图处理为小批次训练的格式
    inner_idx = torch.nonzero(node_dict['inner_node'],as_tuple=True)[0]
    graph = dgl.to_block(dgl.in_subgraph(graph, inner_idx), inner_idx)

    node_dict.pop('train_mask')
    node_dict.pop('val_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)


    for epoch in range(100):
        model.train()
        logits = model(graph, feat)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc = evaluate(graph, feat, labels, val_mask, model, device)        #模型评估
        
        #如果有必要，可以进行中间结果的输出。根据中间结果可以选择性的保存模型或者选择最终进行测试的模型。
        if rank==0 : print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='distGNN')
    parser.add_argument("--n-partitions", "--n_partitions", type=int, default=1,
                        help="the number of partitions")
    
    args = parser.parse_args()

    num_gpus = args.n_partitions
    
    #载入reddit数据集
    from dgl.data import RedditDataset
    data = RedditDataset(raw_dir='./dataset/')
    g = data[0]
    n_feat = g.ndata['feat'].shape[1]
    n_hidden = 128
    n_class = g.ndata['label'].max().item() + 1
    n_layers = 2

    #划分数据集(*@\label{line:part_dataset_l}@*)
    graph_dir = 'partitions/' + graph_name + '/'
    part_config = graph_dir + graph_name + '.json'
    if not os.path.exists(part_config):
        dgl.distributed.partition_graph(g, graph_name, num_gpus, 'partitions/' + graph_name, balance_edges=True)

    #启动训练进程
    mp.spawn(run, args=(list(range(num_gpus)), n_feat, n_hidden, n_class, n_layers), nprocs=num_gpus)
