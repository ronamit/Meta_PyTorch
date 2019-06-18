
#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml

from __future__ import absolute_import, division, print_function

import timeit

from Models.deterministic_models import get_model
from Utils import common as cmn, data_gen
from Utils.common import grad_step, correct_rate, write_to_log, count_correct
from Utils.Losses import get_loss_func
from torch.optim import SGD
from Single_Task.learn_single_standard import run_test

def run_learning(task_data, meta_model, prm, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Loss criterion
    loss_criterion = get_loss_func(prm)

    # Create model for task:
    task_model = get_model(prm)
    
    #  Load initial point from meta-parameters:
    task_model.load_state_dict(meta_model.state_dict())

    # The data-sets of the new task:
    train_loader = task_data['train']
    test_loader = task_data['test']
    n_train_samples = len(train_loader.dataset)
    n_batches = len(train_loader)

    #  Get task optimizer:
    task_optimizer = SGD(task_model.parameters(), lr=prm.alpha)
    # In meta-testing, use SGD with step-size alpha

    # -------------------------------------------------------------------------------------------
    #  Learning  function
    # -------------------------------------------------------------------------------------------

    def run_meta_test_learning(task_model, train_loader):      

        task_model.train()
        train_loader_iter = iter(train_loader)

        # Gradient steps (training) loop
        for i_grad_step in range(prm.n_meta_test_grad_steps):
            # get batch:
            batch_data = data_gen.get_next_batch_cyclic(train_loader_iter, train_loader)
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            batch_size = inputs.shape[0]

            # Calculate empirical loss:
            outputs = task_model(inputs)
            task_objective = (1 / batch_size) * loss_criterion(outputs, targets)

            # Take gradient step with the task weights:
            grad_step(task_objective, task_optimizer)

        # end gradient step loop

        return task_model



    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    if verbose == 1:
        write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    task_model = run_meta_test_learning(task_model, train_loader)

    # Test:
    test_acc, _ = run_test(task_model, test_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm, verbose=verbose)

    test_err = 1 - test_acc
    return test_err, task_model
