import numpy as np
from values import delta_x, delta_y, delta_t, data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from model import LSTMPredictor




if __name__=="__main__":
    trace_num = data.shape[0] * data.shape[1]
    n=data.shape[2]
    data=data.reshape((trace_num, n))
    scaler=MinMaxScaler()
    future = 50
    data=scaler.fit_transform(data[:, :int(n/2)+future])
    halfdata=data[:, :int(n/2)]
    checktarget=data[:, int(n/2):]

    train_input = torch.from_numpy(halfdata[400:501, :-1])
    train_target = torch.from_numpy(halfdata[400:501, 1:])
    test_input = torch.from_numpy(halfdata[560:565, :-1])
    test_target = torch.from_numpy(halfdata[560:565, 1:])
    target_check=torch.from_numpy(checktarget[560:565, :future])
    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.05)
    n_steps = 501
    losses=[]
    was=np.empty(int(n/2)+future)
    has=np.empty(int(n/2)+future)
    for i in range(n_steps):
        print("Step: ", i)


        def closure():
            optimizer.zero_grad()
            y = model(train_input)
            loss = criterion(y, train_target)
            print("loss: ", loss.item())
            loss.backward()
            return loss


        optimizer.step(closure)
        with torch.no_grad():

            pred = model(test_input, future)
            loss = criterion(pred[:, :-future], test_target)
            print("test loss: ", loss.item())
            losses.append(loss)
            if i % 10 == 0:
                y = pred.detach().numpy()
                plt.figure(figsize=(12, 6))
                plt.title(f"Step: {i + 1}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)


                def draw(y_i, t_check, t_target, t_input):
                    global was
                    was=np.concatenate((t_input.numpy()[:1], t_target.numpy(), t_check.numpy()))
                    was=scaler.inverse_transform(was.reshape(1, -1))
                    global has
                    has=np.concatenate((t_input.numpy()[:1], y_i))
                    has = scaler.inverse_transform(has.reshape(1, -1))
                    n = t_target.shape[0]
                    plt.plot(np.arange(n), was[0, 1:-future], "r", linewidth=2)
                    plt.plot(np.arange(n, n + future), was[0, -future:], "r" + ":", linewidth=2)
                    plt.plot(np.arange(n + future), has[0, 1:], "g" + ":", linewidth=2)
                    print("Mse loss for predicted points=", criterion(torch.tensor(was[0, -future:]), torch.tensor(has[0, -future:])).item())



                draw(y[0], target_check[0], test_target[0], test_input[0])
                plt.savefig(f"predict{i + 1}.pdf")
                plt.close()




    plt.figure(figsize=(12, 6))
    plt.title("Loss Plot")
    plt.xlabel("steps")
    plt.ylabel("losses")
    losses=[l for l in losses if l<1]
    plt.plot(np.arange(len(losses)), losses)
    
    plt.savefig("loss_plot.pdf")
    plt.close()
    with torch.no_grad():
        ls=nn.L1Loss()
        first=was[0, -future:]
        second=has[0, -future:]
        print(first)
        print(second)
        errors=[]
        for i in range(1, first.shape[0]):
            errors.append(ls(torch.tensor(first[:i]), torch.tensor(second[:i])).item())

        plt.figure(figsize=(12, 6))
        plt.title("Mininmum Absolute errors")
        plt.xlabel("points predicted")
        plt.ylabel("errors")
        plt.plot(np.arange(len(errors)), errors)

        plt.savefig("errors_points_graph.pdf")
        plt.close()




