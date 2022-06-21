def get_POP_conf_dict(experiment):
    d = {
        #
        'num_epochs': experiment.get_parameter("num_epochs"),
        'early_stop': experiment.get_parameter("early_stop"),
        'learning_rate': experiment.get_parameter("learning_rate"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'patience': experiment.get_parameter("patience"),
    }
    return d

def get_AE_conf_dict(experiment):
    d = {
        #
        'act': experiment.get_parameter("act"),
        'hidden_dim': experiment.get_parameter("hidden_dim"),
        'sparse_normalization': experiment.get_parameter("sparse_normalization"),
        #
        'learning_rate': experiment.get_parameter("learning_rate"),
        'weight_decay' : experiment.get_parameter("weight_decay"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'early_stop': experiment.get_parameter("early_stop"),
        'num_epochs': experiment.get_parameter("num_epochs"),
        'patience': experiment.get_parameter("patience")
    }
    return d

def get_model_name(experiment):
    return experiment.get_parameter("model_name")

def get_VAE_conf_dict(experiment):
    d = get_AE_conf_dict(experiment)
    vae = {
        'anneal_cap': experiment.get_parameter("anneal_cap"),
        'total_anneal_steps': experiment.get_parameter("total_anneal_steps"),
    }
    d.update(vae)
    return d 

def get_VAEsigma_conf_dict(experiment):
    d = get_AE_conf_dict(experiment)
    vae = {
        'decoder_bias': experiment.get_parameter("decoder_bias"),
        'training_type': experiment.get_parameter("training_type"),
        'global_variance': experiment.get_parameter("global_variance")
    }
    d.update(vae)
    return d 

def get_VAEmultilayer_conf_dict(experiment):
    d = {
        #
        'act': experiment.get_parameter("act"),
        'hidden_dim': experiment.get_parameter("hidden_dim"),
        'sparse_normalization': experiment.get_parameter("sparse_normalization"),
        'dropout_ratio': experiment.get_parameter("dropout_ratio"),
        #
        'learning_rate': experiment.get_parameter("learning_rate"),
        'weight_decay' : experiment.get_parameter("weight_decay"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'early_stop': experiment.get_parameter("early_stop"),
        'num_epochs': experiment.get_parameter("num_epochs"),
        'patience': experiment.get_parameter("patience"),
        'anneal_cap': experiment.get_parameter("anneal_cap"),
        'total_anneal_steps': experiment.get_parameter("total_anneal_steps"),
        'weighted_recon': experiment.get_parameter("weighted_recon")
    }
    return d

def get_VAEsinglelayer_conf_dict(experiment):
    d = {
        'act': experiment.get_parameter("act"),
        #
        'hidden_dim': experiment.get_parameter("hidden_dim"),
        'sparse_normalization': experiment.get_parameter("sparse_normalization"),
        'dropout_ratio': experiment.get_parameter("dropout_ratio"),
        #
        'learning_rate': experiment.get_parameter("learning_rate"),
        'weight_decay' : experiment.get_parameter("weight_decay"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'early_stop': experiment.get_parameter("early_stop"),
        'num_epochs': experiment.get_parameter("num_epochs"),
        'patience': experiment.get_parameter("patience"),
        'anneal_cap': experiment.get_parameter("anneal_cap"),
        'total_anneal_steps': experiment.get_parameter("total_anneal_steps"),
    }
    return d

def get_QVAE_conf_dict(experiment):
    d = {
        'act': experiment.get_parameter("act"),
        #
        'hidden_dim': experiment.get_parameter("hidden_dim"),
        'sparse_normalization': experiment.get_parameter("sparse_normalization"),
        'dropout_ratio': experiment.get_parameter("dropout_ratio"),   # not being tuned for now
        #
        'learning_rate': experiment.get_parameter("learning_rate"),
        'weight_decay': experiment.get_parameter("weight_decay"),
        'max_log_var': experiment.get_parameter("max_log_var"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'early_stop': experiment.get_parameter("early_stop"),
        'num_epochs': experiment.get_parameter("num_epochs"),
        'patience': experiment.get_parameter("patience"),
        'anneal_cap': experiment.get_parameter("anneal_cap"),
        'total_anneal_steps': experiment.get_parameter("total_anneal_steps"),
    }
    return d

def get_VAEsigmamultilayer_conf_dict(experiment):
    d = {
        #
        'hidden_dim': experiment.get_parameter("hidden_dim"),
        'sparse_normalization': experiment.get_parameter("sparse_normalization"),
        'dropout_ratio': experiment.get_parameter("dropout_ratio"),
        #
        'learning_rate': experiment.get_parameter("learning_rate"),
        'weight_decay' : experiment.get_parameter("weight_decay"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'early_stop': experiment.get_parameter("early_stop"),
        'num_epochs': experiment.get_parameter("num_epochs"),
        'patience': experiment.get_parameter("patience"),
        'decoder_bias': experiment.get_parameter("decoder_bias"),
        'training_type': experiment.get_parameter("training_type"),
        'global_variance': experiment.get_parameter("global_variance")
    }
    return d

def get_VAEcontrast_conf_dict(experiment):
    d = {
        'act': experiment.get_parameter("act"),
        #
        'hidden_dim': experiment.get_parameter("hidden_dim"),
        'sparse_normalization': experiment.get_parameter("sparse_normalization"),
        'dropout_ratio': experiment.get_parameter("dropout_ratio"),
        'pos_uk_num': experiment.get_parameter("pos_uk_num"),
        'neg_uk_num': experiment.get_parameter("neg_uk_num"),
        'pos_kk_num': experiment.get_parameter("pos_kk_num"),
        'neg_kk_num': experiment.get_parameter("neg_kk_num"),
        'kernel_method': experiment.get_parameter("kernel_method"),
        'temperature_tau_u': experiment.get_parameter("temperature_tau_u"),
        'temperature_tau_k': experiment.get_parameter("temperature_tau_k"),
        #
        'learning_rate': experiment.get_parameter("learning_rate"),
        'weight_decay' : experiment.get_parameter("weight_decay"),
        'batch_size': experiment.get_parameter("batch_size"),
        'test_batch_size': experiment.get_parameter("test_batch_size"),
        'early_stop': experiment.get_parameter("early_stop"),
        'num_epochs': experiment.get_parameter("num_epochs"),
        'patience': experiment.get_parameter("patience"),
        'anneal_cap': experiment.get_parameter("anneal_cap"),
        'total_anneal_steps': experiment.get_parameter("total_anneal_steps"),
        'use_default_hp': experiment.get_parameter("use_default_hp"),
        'hp_contrastive_u': experiment.get_parameter("hp_contrastive_u"),
        'weighted_recon': experiment.get_parameter("weighted_recon")
        # 'hp_contrastive_k': experiment.get_parameter("hp_contrastive_k")
        # 'max_var': experiment.get_parameter("max_var")
    }
    return d

conf_dict_generator = {
    'POP': get_POP_conf_dict,
    'AE': get_AE_conf_dict,
    'VAE': get_VAE_conf_dict,
    'VAEsigma': get_VAEsigma_conf_dict,
    'VAEcontrast': get_VAEcontrast_conf_dict,
    'VAEcontrast_multilayer': get_VAEcontrast_conf_dict,
    'VAEmultilayer_contrast': get_VAEcontrast_conf_dict,
    'VAEcontrast_multilayer_wcontext': get_VAEcontrast_conf_dict,
    'VAEsinglelayer': get_VAEsinglelayer_conf_dict,
    'VAEmultilayer': get_VAEmultilayer_conf_dict,
    'VAEsigmamultilayer': get_VAEsigmamultilayer_conf_dict,
    'QVAE': get_QVAE_conf_dict,
    'QVAE_multi': get_QVAE_conf_dict,   # save hps with single layer's settings
}