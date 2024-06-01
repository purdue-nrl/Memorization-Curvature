"""
@author: Anonymized
@copyright: Anonymized
"""

import os
import multiprocessing

def main():
    import argparse
    import torch
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from torchvision.models import resnet18
    import random
    import numpy as np
    import logging
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=200,            type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='imagenet',     type=str,       help='Set dataset to use')
    parser.add_argument('--test',                   default=False,          type=str2bool,  help='Calculate curvature on Test Set')
    parser.add_argument('--lr',                     default=0.1,            type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--momentum', '--m',        default=0.9,            type=float,     help='Momentum')
    parser.add_argument('--weight-decay', '--wd',   default=0,              type=float,     metavar='W', help='Weight decay (default: 0)')
    parser.add_argument('--h',                      default=1e-4,           type=float,     help='h for curvature calculation')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=512,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=512,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.00,           type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=False,          type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=False,          type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=True,           type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--parallel',               default=True,           type=str2bool,  help='Device in  parallel')

    global args
    args = parser.parse_args()
    args.arch = 'resnet18'

    # Reproducibility settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '4'
 
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        version_list = list(map(float, torch.__version__.split(".")))
        if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
            torch.set_deterministic(True)
        else:
            torch.use_deterministic_algorithms(True)
    except:
        torch.use_deterministic_algorithms(True)

    # Setup right device to run on
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger(f'Compute Logger')
    logger.setLevel(logging.INFO)

    model_name = f'{args.dataset.lower()}_{args.arch}_{args.suffix}'
    handler = logging.FileHandler(os.path.join('./logs', f'score_{model_name}_curv_scorer_gpu9_he-3.log'))
    formatter = logging.Formatter(
        fmt=f'%(asctime)s %(levelname)-8s %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info(args)

    index = torch.load("./curv_scores/data_index.pt")
    dataset_len = len(index)
    logger.info("Loaded index")

    # Use the following transform for training and testing
    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        index=index,
        random_seed=args.random_seed,
        distributed=False)

    # Instantiate model 
    net = resnet18()
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2])

    criterion = torch.nn.CrossEntropyLoss()

    def get_regularized_curvature_for_batch(batch_data, batch_labels, h=1e-3, niter=10, temp=1):
        num_samples = batch_data.shape[0]
        net.eval()
        regr = torch.zeros(num_samples)
        eigs = torch.zeros(num_samples)
        for _ in range(niter):
            v = torch.randint_like(batch_data, high=2).cuda()
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            v = h * (v + 1e-7)

            batch_data.requires_grad_()
            outputs_pos = net(batch_data + v)
            outputs_orig = net(batch_data)
            loss_pos = criterion(outputs_pos / temp, batch_labels)
            loss_orig = criterion(outputs_orig / temp, batch_labels)
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data )[0]

            regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            eigs += torch.diag(torch.matmul(v.reshape(num_samples,-1), grad_diff.reshape(num_samples,-1).T)).cpu().detach()
            net.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()

        curv_estimate = eigs / niter
        regr_estimate = regr / niter
        return curv_estimate, regr_estimate

    def score_true_labels_and_save(epoch, test, logger, model_name):
        scores = torch.zeros((dataset_len))
        regr_score = torch.zeros_like(scores)
        labels = torch.zeros_like(scores, dtype=torch.long)
        net.eval()
        total = 0
        dataloader = dataset.train_loader if not test else dataset.test_loader
        for (inputs, targets) in tqdm(dataloader):
            start_idx = total
            stop_idx = total + len(targets)
            idxs = index[start_idx:stop_idx]
            total = stop_idx

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs.requires_grad = True
            net.zero_grad()

            curv_estimate, regr_estimate = get_regularized_curvature_for_batch(inputs, targets, h=args.h, niter=10)
            scores[idxs] = curv_estimate.detach().clone().cpu()
            regr_score[idxs] = regr_estimate.detach().clone().cpu()
            labels[idxs] = targets.cpu().detach()

        scores_file_name = f"scores_{epoch}_{model_name}_{args.h}.pt" if not test else f"scores_{epoch}_{model_name}_{args.h}_test.pt"
        regr_file_name = f"regr_scores_{epoch}_{model_name}_{args.h}.pt" if not test else f"regr_scores_{epoch}_{model_name}_{args.h}_test.pt"
        labels_file_name = f"true_labels{epoch}_{model_name}_{args.h}.pt" if not test else f"true_labels{epoch}_{model_name}_{args.h}_test.pt"
        logger.info(f"Saving {scores_file_name}")
        torch.save(scores, os.path.join('curv_scores', scores_file_name))
        torch.save(regr_score, os.path.join('curv_scores', regr_file_name))
        torch.save(labels, os.path.join('curv_scores', labels_file_name))
        return


    for epoch in range(0, 200, 4):
        logger.info(f'Loading model for epoch {epoch}')
        
        model_state = torch.load('./curv_scores/'+ args.dataset.lower()+'_hiker_wd0/' + model_name + f'{epoch}.ckpt')
        if args.parallel:
            net.module.load_state_dict(model_state)
        else:
            net.load_state_dict(model_state)

        net.to(device)
        test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
        logger.info(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

        # Calculate curvature score
        score_true_labels_and_save(epoch, args.test, logger, model_name)

    if args.parallel:
        destroy_process_group()

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()