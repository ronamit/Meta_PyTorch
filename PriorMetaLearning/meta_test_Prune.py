
from __future__ import absolute_import, division, print_function

import timeit
from collections import OrderedDict
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import run_eval_Bayes
from Utils.complexity_terms import get_task_complexity
from Utils.common import grad_step, count_correct, write_to_log
from Utils.Losses import get_loss_func
import Models.deterministic_models as func_models
from Models.stochastic_layers import StochasticLayer
import torch
# from Single_Task.learn_single_standard import run_test

def run_learning(task_data, prior_model, prm, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_func(prm)

    # Create  model for the new task:
    model = func_models.get_model(prm)

    # TODO: how to initialize?

    # The data-sets of the new task:
    train_loader = task_data['train']
    test_loader = task_data['test']
    n_train_samples = len(train_loader.dataset)
    n_batches = len(train_loader)

    #  Get optimizer:
    optimizer = optim_func(model.parameters(), **optim_args)

    # Create Pruned NN:

    # no need for gradients of prior
    for param in prior_model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500
        prior_layers_list = OrderedDict((name, layer) for (name, layer) in prior_model.named_children()
            if isinstance(layer, StochasticLayer))
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            # get batch data:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            batch_size = inputs.shape[0]

            net_weights = OrderedDict()

            for (name, param) in model.named_parameters():
                layer_name = name.split('.')[-2]
                weight_type = name.split('.')[-1]  # bias or weight
                prior_layer = prior_layers_list[layer_name]
                if weight_type == 'weight':
                    mu = prior_layer.w_mu
                    log_var = prior_layer.w_log_var
                elif weight_type == 'bias':
                    mu = prior_layer.b_mu
                    log_var = prior_layer.b_log_var
                else:
                    raise ValueError('Unrecognized weight_type')
                std = torch.exp(0.5 * log_var)
                param_tuned = mu + std * param
                #TODO: prune lower precntile of std values qa
                net_weights.update({name: param_tuned})

            # Calculate empirical loss:
            outputs = model(inputs, net_weights)
            avg_empiric_loss = (1 / batch_size) * loss_criterion(outputs, targets)

            correct_count = count_correct(outputs, targets)
            sample_count = inputs.size(0)

            total_objective = avg_empiric_loss

            # Take gradient step with the posterior:
            grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)


            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_count / sample_count
                print(cmn.status_string(i_epoch, prm.n_meta_test_epochs, batch_idx, n_batches, batch_acc, total_objective.item()) +
                      ' Empiric Loss: {:.4}\t'.
                      format(avg_empiric_loss.item()))
        # end batch loop
        return net_weights
    # end run_train_epoch()


    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    if verbose == 1:
        write_to_log('Total number of steps: {}'.format(n_batches * prm.n_meta_test_epochs), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    prm.n_meta_test_epochs = 10 # TODO: fix
    for i_epoch in range(prm.n_meta_test_epochs):
        net_weights = run_train_epoch(i_epoch)


    # Test:
    # TODO: set net_weights as the paramters of the model and use  from Single_Task.learn_single_standard import run_test
    test_acc, _ = run_test(model, test_loader, loss_criterion, prm, net_weights)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm, result_name=prm.test_type, verbose=verbose)

    test_err = 1 - test_acc
    return test_err, model




# -------------------------------------------------------------------------------------------
#  Test evaluation function Aaa
# --------------------------------------------------------------------------------------------
def run_test(model, test_loader, loss_criterion, prm, net_weights):
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        batch_size = inputs.shape[0]
        outputs = model(inputs, net_weights)
        test_loss += (1 / batch_size) * loss_criterion(outputs, targets).item()  # sum the mean loss in batch
        n_correct += count_correct(outputs, targets)

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    test_loss = test_loss / n_test_batches
    test_acc = n_correct / n_test_samples
    print('\n Standard learning: test loss: {:.4}, test err: {:.3} ( {}/{})\n'.format(
        test_loss, 1-test_acc, n_correct, n_test_samples))
    return test_acc, test_loss
