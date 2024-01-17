from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/adminuser/extra_hdd_1/data/got10k_lmdb'
    settings.got10k_path = '/home/adminuser/extra_hdd_1/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/adminuser/extra_hdd_1/data/lasot_lmdb'
    settings.lasot_path = '/home/adminuser/extra_hdd_1/data/lasot'
    settings.network_path = '/home/adminuser/Minho/TETrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/adminuser/extra_hdd_1/data/nfs'
    settings.otb_path = '/home/adminuser/extra_hdd_1/data/OTB2015'
    settings.prj_dir = '/home/adminuser/Minho/TETrack'
    settings.result_plot_path = '/home/adminuser/Minho/TETrack/output/test/result_plots'
    settings.results_path = '/home/adminuser/Minho/TETrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/adminuser/Minho/TETrack/output'
    settings.segmentation_path = '/home/adminuser/Minho/TETrack/output/test/segmentation_results'
    settings.tc128_path = '/home/adminuser/extra_hdd_1/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/adminuser/extra_hdd_1/data/trackingnet'
    settings.uav_path = '/home/adminuser/extra_hdd_1/data/UAV123'
    settings.vot_path = '/home/adminuser/extra_hdd_1/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

