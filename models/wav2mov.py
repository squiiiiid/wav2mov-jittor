class Wav2MovTemplate(TemplateModel):
    def __init__(self, hparams, config, logger):
        super().__init__()
        self.hparams = hparams
        self.config = config
        self.logger = logger
        self.accumulation_steps = self.hparams['data']['batch_size'] // self.hparams['data']['mini_batch_size']
        self.set_device()

        self.init_models()
        self.init_optims()
        self.init_obj_functions()
        self.init_schedulers()
        self.scaler = amp.GradScaler()

    def set_device(self):
        device = self.hparams['device']
        if device == 'cuda':
            device = 'cpu' if not torch.cuda.is_available() else device
        self.device = torch.device(device)

    def init_models(self):
        self.gen = Generator(self.hparams['gen'])
        self.seq_disc = SequenceDiscriminator(self.hparams['disc']['sequence_disc'])
        # self.id_disc = PatchDiscriminator(self.hparams['disc']['patch_disc'])
        self.id_disc = IdentityDiscriminator(self.hparams['disc']['identity_disc'])
        self.sync_disc = SyncDiscriminator(self.hparams['disc']['sync_disc'],self.config)
        init_net(self.gen)
        init_net(self.seq_disc)
        init_net(self.id_disc)
        init_net(self.sync_disc)
        self.set_train_mode()

    def init_optims(self):
        self.optim_gen = self.gen.get_optimizer()
        self.optim_seq_disc = self.seq_disc.get_optimizer()
        self.optim_id_disc = self.id_disc.get_optimizer()
        self.optim_sync_disc = self.sync_disc.get_optimizer()

    def init_obj_functions(self):
        self.criterion_gan = GANLoss(self.device)
        self.criterion_L1 = L1_Loss()
        self.criterion_sync = SyncLoss(self.device)

    def init_schedulers(self):
        gen_step_size = self.hparams['scheduler']['gen']['step_size']
        discs_step_size = self.hparams['scheduler']['discs']['step_size']
        gen_gamma = self.hparams['scheduler']['gen']['gamma']
        discs_gamma = self.hparams['scheduler']['discs']['gamma']