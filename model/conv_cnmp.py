
import torch
import torch.nn as nn

r"""
                   input_size   output_size kernel_size stride      padding     operation
    conv_layers = [[x_11,       x_12,       x_13,       x_14,       x_15], ---> conv_1
                   [x_21,       x_22,       x_23,       x_24,       x_25], ---> conv_2
                   [x_31,       x_32,       x_33,       x_34,       x_35], ---> conv_3
                   .            .           .           .           .           .
                   .            .           .           .           .           .
                   .            .           .           .           .           .
                   [x_n1,       x_n2,       x_n3,       x_n4,       x_n5]] ---> conv_n

    x_i2 = x_(i+1)1 for all i in {1, ..., n-1}

    conv_image_height = h_01
    conv_image_width = w_01

    pool_kernel_size = pk
    pool_kernel_stride = ps
    pool_padding = 0
    batch_size = N
    
    ------------------------------------------------------------------------------------------------------------------------------------------

    general formula for convolution dimension:
    next_layer_size = (layer_size - conv_kernel_size + 2*conv_padding) // conv_stride + 1

    for this model:
    h_i0 = (h_(i-1)1 - x_i3 + 2*x_i5) // x_i4 + 1 for all i in {1, ..., n}
    w_i0 = (w_(i-1)1 - x_i3 + 2*x_i5) // x_i4 + 1 for all i in {1, ..., n}

    ------------------------------------------------------------------------------------------------------------------------------------------

    general formula for pooling dimension:
    next_layer_size = (layer_size - pool_kernel_size + 2*pool_padding) // pool_stride + 1

    for this model:
    h_i1 = (h_i0 - pk) // ps + 1 for all i in {1, ..., n-1} // no pooling after the last convolution
    w_i1 = (w_i0 - pk) // ps + 1 for all i in {1, ..., n-1} // no pooling after the last convolution

    ------------------------------------------------------------------------------------------------------------------------------------------

    formula of flatten:
    Tensor(a_1, a_2, a_3) -> Tensor(a_1*a_2*a_3)

    linear_output_sizes = [l_1, l_2, ..., l_m]

    ------------------------------------------------------------ FLOW OF DIMENSION ------------------------------------------------------------

                        N,      x_11,           h_01,                                               w_01
    relu(conv_1)        N,      x_12,           h_10 = (h_01 - x_13 + 2*x_15) // x_14 + 1,          w_10 = (w_01 - x_13 + 2*x_15) // x_14 + 1
    pool                N,      x_12,           h_11 = (h_10 - pk) // ps + 1,                       w_11 = (w_10 - pk) // ps + 1
    relu(conv_2)        N,      x_22,           h_20 = (h_11 - x_23 + 2*x_25) // x_24 + 1,          w_20 = (w_11 - x_23 + 2*x_25) // x_24 + 1
    pool                N,      x_22,           h_21 = (h_20 - pk) // ps + 1,                       w_21 = (w_20 - pk) // ps + 1
    .                   .       .               .                                                   .
    .                   .       .               .                                                   .
    .                   .       .               .                                                   .
    relu(conv_n)        N,      x_n2,           h_n0 = (h_(n-1)1 - x_n3 + 2*x_n5) // x_n4 + 1,      w_n0 = (w_(n-1)1 - x_n3 + 2*x_n5) // x_n4 + 1
    flatten             N,      x_n2*h_n0*w_n0
    relu(linear_1)      N,      l_1
    relu(linear_2)      N,      l_2
    .                   .       .
    .                   .       .
    .                   .       .
    linear_m            N,      l_m

    """
class ConvCNMP(nn.Module):                # input size of default conv1 is 3 because it is (R,G,B)
    def __init__(self, conv_layers: list = [[3,256,3,1,0], [256,64,3,1,0], [64,16,3,1,0]], conv_image_height:int = 32, 
            conv_image_width: int = 32, pool_kernel_size: int = 2, pool_stride: int = 2, linear_output_sizes: list = [32,10], 
            cnmp_input_dim: int = 1, cnmp_encoder_hidden_dims: list = [256,256,256], cnmp_decoder_hidden_dims: list = [256,256,256], 
            cnmp_output_dim: int = 1, cnmp_max_obs: int = 10, cnmp_max_tar: int = 10, batch_size: int = 64):
        
        # ----------------------------------------------------- PARAMETER CHECKS ----------------------------------------------------- #
        # region parameter checks
        if len(conv_layers) <= 0:
            raise Exception("At least one convolution layer must exist")

        for index, layer in enumerate(conv_layers):
            if len(layer) != 5:
                raise Exception("Every convolution layer must have exactly 5 parameters: input_size, output_size, kernel_size, stride, padding\nLayer %d has %d parameter(s)" % (index, len(layer)))
            if layer[0] <= 0 or layer[1] <= 0 or layer[2] <= 0 or layer[3] <= 0:
                raise Exception("Convolution Layer %d = %s: input size, output size, kernel size, or stride cannot be non-positive" % (index, layer))
            if layer[4] < 0:
                raise Exception("Convolution Layer %d = %s: padding cannot be negative" % (index, layer))

        if conv_image_height <= 0:
            raise Exception("Convolution image height must be positive")

        if conv_image_width <= 0:
            raise Exception("Convolution image width must be positive")
        
        if pool_kernel_size <= 0:
            raise Exception("Convolution pool kernel size must be positive")
        
        if pool_stride <= 0:
            raise Exception("Convolution pool stride must be positive")
        
        if len(linear_output_sizes) <= 0:
            raise Exception("At least one linear layer must exist")

        for index, size in enumerate(linear_output_sizes):
            if size <= 0:
                raise Exception("Linear Layer %d has non-positive output size: %d" % (index, size))
        
        if cnmp_input_dim <= 0:
            raise Exception("CNMP input dimension must be positive")

        if len(cnmp_encoder_hidden_dims) <= 0:
            raise Exception("At least one CNMP encoder layer must exist")
        
        for index, dim in enumerate(cnmp_encoder_hidden_dims):
            if dim <= 0:
                raise Exception("CNMP Encoder Layer %d has non-positive dimension: %d" % (index, dim))
            
        if len(cnmp_decoder_hidden_dims) == 0:
            raise Exception("At least one CNMP decoder layer must exist")
        
        for index, dim in enumerate(cnmp_decoder_hidden_dims):
            if dim <= 0:
                raise Exception("CNMP Decoder Layer %d has non-positive dimension: %d" % (index, dim))
            
        if cnmp_output_dim <= 0:
            raise Exception("CNMP output dimension must be positive")
        
        if cnmp_max_obs <= 0:
            raise Exception("CNMP max number of observations must be positive")
        
        if cnmp_max_tar <= 0:
            raise Exception("CNMP max number of targets must be positive")
        
        if batch_size <= 0:
            raise Exception("Batch size must be positive")
        # endregion
        # ---------------------------------------------------------------------------------------------------------------------------- #

        super(ConvCNMP, self).__init__()
        self.conv_layers = conv_layers
        self.conv_num_layers = len(conv_layers)
        self.conv_image_height = conv_image_height
        self.conv_image_width = conv_image_width

        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

        self.linear_output_sizes = linear_output_sizes
        self.num_linear_layers = len(linear_output_sizes)

        self.cnmp_input_dim = cnmp_input_dim
        self.cnmp_encoder_hidden_dims = cnmp_encoder_hidden_dims
        self.cnmp_encoder_num_layers = len(cnmp_encoder_hidden_dims)
        self.cnmp_decoder_hidden_dims = cnmp_decoder_hidden_dims
        self.cnmp_decoder_num_layers = len(cnmp_decoder_hidden_dims)
        self.cnmp_output_dim = cnmp_output_dim
        self.cnmp_max_obs = cnmp_max_obs
        self.cnmp_max_tar = cnmp_max_tar
        
        self.batch_size = batch_size

        conv_sequence = []
        dimension = [conv_layers[0][0], conv_image_height, conv_image_width]
        for i in range(self.conv_num_layers):
            dimension = [conv_layers[i][1], (dimension[1] - conv_layers[i][2] + 2*conv_layers[i][4]) // conv_layers[i][3] + 1, 
                                            (dimension[2] - conv_layers[i][2] + 2*conv_layers[i][4]) // conv_layers[i][3] + 1]
            conv_sequence.append(nn.Conv2d(*conv_layers[i]))
            conv_sequence.append(nn.ReLU())
            if i < self.conv_num_layers-1:
                dimension = [conv_layers[i][1], (dimension[1] - pool_kernel_size) // pool_stride + 1,
                                                (dimension[2] - pool_kernel_size) // pool_stride + 1]
                conv_sequence.append(nn.MaxPool2d(pool_kernel_size, pool_stride))
        
        dimension = dimension[0] * dimension[1] * dimension[2]
        conv_sequence.append(nn.Flatten(1))

        for i in range(self.num_linear_layers-1):
            conv_sequence.append(nn.Linear(dimension, linear_output_sizes[i]))
            conv_sequence.append(nn.ReLU())
        conv_sequence.append(nn.Linear(linear_output_sizes[-2], linear_output_sizes[-1]))
        self.conv = nn.Sequential(*conv_sequence)

        encoder_sequence = []
        for i in range(self.cnmp_encoder_num_layers-1):
            if i == 0:
                encoder_sequence.append(nn.Linear(linear_output_sizes[-1] + cnmp_input_dim + cnmp_output_dim, cnmp_encoder_hidden_dims[0]))
            else:
                encoder_sequence.append(nn.Linear(cnmp_encoder_hidden_dims[i-1], cnmp_encoder_hidden_dims[i]))
            encoder_sequence.append(nn.ReLU())
        encoder_sequence.append(nn.Linear(cnmp_encoder_hidden_dims[-2], cnmp_encoder_hidden_dims[-1]))
        self.encoder = nn.Sequential(*encoder_sequence)

        decoder_sequence = [nn.Linear(cnmp_input_dim + cnmp_encoder_hidden_dims[-1], cnmp_decoder_hidden_dims[0])]
        for i in range(1, self.cnmp_decoder_num_layers):
            decoder_sequence.append(nn.ReLU())
            decoder_sequence.append(nn.Linear(cnmp_decoder_hidden_dims[i-1], cnmp_decoder_hidden_dims[i]))
        decoder_sequence.append(nn.ReLU())
        decoder_sequence.append(nn.Linear(cnmp_decoder_hidden_dims[-1], 2*cnmp_output_dim))
        self.decoder = nn.Sequential(*decoder_sequence)

    def forward(self, conv_obs, cnmp_obs, cnmp_tar):
        # conv_obs: (batch_size, 3, 64, 64)
        conv_result = self.conv(conv_obs) # (batch_size, linear_output_sizes[-1])
        conv_result_rep = conv_result.unsqueeze(1).repeat(1, cnmp_obs.shape[1], 1)  # conv_result is repeated to match cnmp_obs. we append same conv_result to each cnmp_obs

        total_obs = torch.cat((cnmp_obs, conv_result_rep), dim=-1)
        
        # cnmp_obs: (batch_size, n_o (<cnmp_max_obs), input_dim+output_dim)
        # cnmp_tar: (batch_size, n_t (<cnmp_max_tar), input_dim)
        encoded_obs = self.encoder(total_obs) # (batch_size, n_o (<cnmp_max_obs), cnmp_encoder_hidden_dims[-1])
        encoded_rep = encoded_obs.mean(dim=1).unsqueeze(1) # (batch_size, 1, cnmp_encoder_hidden_dims[-1])
        repeated_encoded_rep = torch.repeat_interleave(encoded_rep, cnmp_tar.shape[1], dim=1)  # each encoded_rep is repeated to match cnmp_tar
        rep_tar = torch.cat([repeated_encoded_rep, cnmp_tar], dim=-1)
        
        pred = self.decoder(rep_tar)  # (batch_size, n_t (<cnmp_max_tar), 2*cnmp_output_dim)
        return pred

    def loss(self, pred, real):
        # pred: (batch_size, n_t (<cnmp_max_tar), 2*cnmp_output_dim)
        # real: (batch_size, n_t (<cnmp_max_tar), cnmp_output_dim)
        pred_mean = pred[:, :, :self.cnmp_output_dim]
        pred_std = torch.nn.functional.softplus(pred[:, :, self.cnmp_output_dim:]) + 1e-6 # predicted value is std. In comb. with softplus and minor addition to ensure positivity

        pred_dist = torch.distributions.Normal(pred_mean, pred_std)

        return (-pred_dist.log_prob(real)).mean()
    
