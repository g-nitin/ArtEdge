import datetime
import os


class Config:
    data_num = 60_000  # Maximum number of training data
    root = os.path.dirname(os.path.abspath(__file__))

    file_n = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_dir = "./logs/" + file_n
    # ckpt_dir = "./models/" + file_n
    log_dir = os.path.join(root, "..", "logs", file_n)
    ckpt_dir = os.path.join(root, "..", "models", file_n)

    multi_to_multi = True
    test_content_size = 256
    test_style_size = 256
    content_dir = os.path.join(root, "..", "data", "contents")
    style_dir = os.path.join(root, "..", "data", "styles")
    img_dir = os.path.join(root, "..", "data", "outputs")

    # Customs
    hf_weight = 1  # Weight for high-frequency style (0.0 to 1.0+)
    lf_weight = 1  # Weight for low-frequency style (0.0 to 1.0+)
    save_freq_components = False  # Save separate high/low frequency stylized outputs

    model = os.path.join(root, "..", "models", "main.pth")
    vgg_model = os.path.join(root, "..", "models", "vgg_normalised.pth")

    ## basic parameters
    n_iter = 160_000
    batch_size = 8
    lr = 0.0001
    lr_policy = "step"
    lr_decay_iters = 50
    beta1 = 0.0

    # preprocess parameters
    load_size = 512
    crop_size = 512

    # model parameters
    input_nc = 3  # of input image channel
    nf = 64  # of feature map channel after Encoder first layer
    output_nc = 3  # of output image channel
    style_kernel = 3  # size of style kernel

    # Octave Convolution parameters
    alpha_in = 0.5  # input ratio of low-frequency channel
    alpha_out = 0.5  # output ratio of low-frequency channel
    freq_ratio = [1, 1]  # [high, low] ratio at the last layer

    # Loss ratio
    lambda_percept = 1.0
    lambda_perc_cont = 1.0
    lambda_perc_style = 10.0
    lambda_const_style = 5.0

    # Else
    norm = "instance"
    init_type = "normal"
    init_gain = 0.02
    no_dropout = "store_true"
    num_workers = 8
