import argparse
import torch
import models
import os
import torch.nn as nn
import utils


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='./data/id_0514')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=1)
    # UNet Options
    parser.add_argument("--bilinear", action='store_true', default=False)
    parser.add_argument("--light_attention", action='store_true', default=False,
                        help="Whether to use the attention module")

    parser.add_argument("--total_itrs", type=int, default=30e4,
                        help="epoch number (default: 30k)")
    parser.add_argument("--val_interval", type=int, default=3248,
                        help="epoch interval for eval (default: 100)")
    return parser


def validate(opts, model, test_data, test_gt, val_loss=0.0):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for i in range(len(test_data)):
        eval_out = model(test_data[i].reshape(1, 1, 20, 5))
        val_loss += criterion(eval_out.reshape(1, 8), test_gt[i].reshape(1, 8)).detach().cpu().numpy()
    return val_loss/len(test_data)


def train():
    opts = get_argparser().parse_args()

    # Set up datasets
    train_dataset, train_gt, test_dataset, test_gt = utils.pre_data(opts.data_root, opts.num_classes)

    # Set up device
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model
    model = models.UNet_LA(n_channels=opts.num_channels, n_classes=opts.num_classes, bilinear=opts.bilinear,
                        light_attention=opts.light_attention)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    # Set up loss
    criterion = nn.CrossEntropyLoss(reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for i in range(len(train_dataset)):
            optimizer.zero_grad()
            outputs = model(train_dataset[i].reshape(1, 1, 20, 5))
            loss = criterion(outputs.reshape(1, 8), train_gt[i].reshape(1, 8))
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            cur_itrs += 1
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest.pth')
                print("validation...")
                model.eval()
                val_loss = validate(opts, model, test_data=test_dataset, test_gt=test_gt)
                if val_loss < best_score:  # save best model
                    best_score = val_loss
                    save_ckpt('checkpoints/best.pth')
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    train()