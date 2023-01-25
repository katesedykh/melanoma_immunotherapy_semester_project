from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler
from torch_geometric.loader import DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_base, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels_base*8)
        self.conv2 = GCNConv(hidden_channels_base*8, hidden_channels_base*4)
        self.conv3 = GCNConv(hidden_channels_base*4, hidden_channels_base)
        self.lin = Linear(hidden_channels_base, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch) 

        x = self.lin(x)
        
        return x


def plot_roc_curve(true_y, y_prob, fold, AUC):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    label="fold "+str(fold+1)+", AUC=", str(AUC)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
def make_plot(history):
    fig = plt.figure(figsize=(15, 15))

    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

    ax1.plot(range(num_epochs), history['train_acc'][1], label='train')
    ax2.plot(range(num_epochs), history['train_acc'][2])
    ax3.plot(range(num_epochs), history['train_acc'][3])
    ax4.plot(range(num_epochs), history['train_acc'][4])
    ax5.plot(range(num_epochs), history['train_acc'][5])

    ax1.plot(range(num_epochs), history['test_acc'][1], label='val')
    ax2.plot(range(num_epochs), history['test_acc'][2])
    ax3.plot(range(num_epochs), history['test_acc'][3])
    ax4.plot(range(num_epochs), history['test_acc'][4])
    ax5.plot(range(num_epochs), history['test_acc'][5])

    ax1.set_title("Fold 1")
    ax2.set_title("Fold 2")
    ax3.set_title("Fold 3")
    ax4.set_title("Fold 4")
    ax5.set_title("Fold 5")


    lines_labels = [ax1.get_legend_handles_labels() ]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper left')

    plt.ylabel("Accuracy, %")
    plt.xlabel("Epochs")
    plt.suptitle("Training and validation accuracy", fontsize=14)
    plt.savefig("train_val_accuracy")


k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)

    
def train_epoch(model,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for data in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output,data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == data.y).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for data in dataloader:
        output = model(data)
        loss=loss_fn(output,data.y)
        valid_loss+=loss.item()
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == data.y).sum().item()

    return valid_loss,val_correct

def main(suffix=''):
    DATA_PER_SLIDE_SAVEDIR = "output/final_features"

    # load data (one type)
    dataset_pyg_path = os.path.join(
        DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix, 'patches'+suffix+'_dataset_pyg.pkl'
        )
    dataset_pyg = pickle.load(open(dataset_pyg_path, 'rb'))
    
    # or load data (combine multiple)
    """
    suffix='_512_rot30'
    dataset_pyg_savepath = os.path.join(
        DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix, 'patches'+suffix+'_dataset_pyg.pkl'
        )
    dataset_pyg_512_rot30 = pickle.load(open(dataset_pyg_savepath, 'rb'))

    suffix='_1024_shift512_rot30'
    dataset_pyg_savepath = os.path.join(
        DATA_PER_SLIDE_SAVEDIR, 'experiment_1'+suffix, 'patches'+suffix+'_dataset_pyg.pkl'
        )
    dataset_pyg_1024_shift512_rot30 = pickle.load(open(dataset_pyg_savepath, 'rb'))

    dataset_pyg.extend(dataset_pyg_512_rot30)
    dataset_pyg.extend(dataset_pyg_1024_shift512_rot30)
    """

    num_epochs = 200
    batch_size = 1
    DROPOUT = 0.1
    criterion = torch.nn.CrossEntropyLoss()

    history = {'train_loss': {}, 'test_loss': {},'train_acc':{},'test_acc':{}, 'roc_val':{}}
    all_AUC = []
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset_pyg)))):

        print('Fold {}'.format(fold + 1))

        history['train_loss'][fold + 1] = []
        history['test_loss'][fold + 1] = []
        history['train_acc'][fold + 1] = []
        history['test_acc'][fold + 1] = []

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset_pyg, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset_pyg, batch_size=batch_size, sampler=test_sampler)


        model = GCN(in_channels=115, hidden_channels_base=8, num_classes=2).double()
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,train_loader,criterion,optimizer)
            test_loss, test_correct=valid_epoch(model,test_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} Training Loss:{:.3f} Test Loss:{:.3f} Training Acc {:.2f} % Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['train_loss'][fold + 1].append(train_loss)
            history['test_loss'][fold + 1].append(test_loss)
            history['train_acc'][fold + 1].append(train_acc)
            history['test_acc'][fold + 1].append(test_acc) 
    
        true_y = []
        pred_y = []
        roc_loader = DataLoader([dataset_pyg[index] for index in val_idx], batch_size=1)
        for data in roc_loader:
            output = model(data)
            true_y.append(data.y.item())
            scores, predictions = torch.max(output.data,1)
            pred_y.append(scores.item())
        AUC=roc_auc_score(true_y, pred_y)    
        print(f'model AUC score: {AUC}')
        all_AUC.append(AUC)
        plot_roc_curve(true_y, pred_y, fold, round(AUC,3))
        plt.legend()
    plt.savefig('figures/fold_'+str(fold+1)+'_roc_GCN_all.png')
    mean_AUC = np.sum(all_AUC)/len(all_AUC)
    print("mean AUC:", mean_AUC)

if __name__ == "__main__":
    main()