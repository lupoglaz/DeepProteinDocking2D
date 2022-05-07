from DeepProteinDocking2D.Models.ReducedSampling.train_brutesimplified_docking import *

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../Datasets/docking_train_400pool'
    validset = '../Datasets/docking_valid_400pool'
    ### testing set
    testset = '../Datasets/docking_test_400pool'
    #########################
    #### initialization torch settings
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    batch_size = 1
    max_size = 1000
    if batch_size > 1:
        raise NotImplementedError()
    train_stream = get_docking_stream(trainset + '.pkl', batch_size, max_size=max_size)
    valid_stream = get_docking_stream(validset + '.pkl', batch_size=1, max_size=max_size)
    test_stream = get_docking_stream(testset + '.pkl', batch_size=1, max_size=max_size)
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))

    ######### Metropolis-Hastings (Monte Carlo) eval on ideal learned energy surface
    train_epochs = 30
    sample_steps = 100
    MC_eval_num_epochs = 5
    sigma_alpha = 3.0
    gamma = 0.5
    experiment = 'BS_pretrain_MC_eval'

    debug=False
    norm='ortho'
    lr = 10 ** -2
    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug, normalization=norm)
    model = SamplingModel(dockingFFT, num_angles=1, IP_MC=True, debug=debug).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ### dummy optimizer to schedule sigma of alpha
    sigma_optimizer = optim.Adam(model.parameters(), lr=sigma_alpha)
    sigma_scheduler = optim.lr_scheduler.ExponentialLR(sigma_optimizer, gamma=gamma)

    ### Train ideal energy surface using BS model
    BruteSimplifiedDockingTrainer(dockingFFT, model, optimizer, experiment, debug=debug).run_trainer(train_epochs, train_stream=train_stream)

    ### Initialiize MC evaluation
    eval_model = SamplingModel(dockingFFT, num_angles=1, sample_steps=sample_steps, IP_MC=True).to(device=0)
    MonteCarloEvaluation = BruteSimplifiedDockingTrainer(dockingFFT, eval_model, optimizer, experiment,
                                                         MC_eval=True, MC_eval_num_epochs=MC_eval_num_epochs,
                                                         sigma_scheduler=sigma_scheduler, sample_buffer_length=sample_buffer_length,
                                                         plotting=False)

    ## Load BS model and do MC eval and plotting
    MonteCarloEvaluation.run_trainer(
        train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=None,
        resume_training=True, resume_epoch=train_epochs,
        sigma_scheduler=sigma_scheduler, sigma_optimizer=sigma_optimizer)
    # Plot loss from current experiment
    IPPlotter(experiment).plot_rmsd_distribution(plot_epoch=train_epochs, show=show, eval_only=True)
