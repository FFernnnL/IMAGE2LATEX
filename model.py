import random
import torch
import torch.nn as nn

# Number of bottlenecks
num_bn = 3
# The depth is half of the actual values in the paper because bottleneck blocks
# are used which contain two convlutional layers
depth = 16
multi_block_depth = depth // 2
growth_rate = 24

n = 256
n_prime = 512
decoder_conv_filters = 256
gru_hidden_size = 256
embedding_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Multi-scale Dense Encoder

    A multi-scale dense encoder with two branches. The first branch produces
    low-resolution annotations, as a regular dense encoder would, and the second branch
    produces high-resolution annotations.
    """

    def __init__(
        self,
        img_channels=1,
        num_in_features=48,
        dropout_rate=0.2,
        checkpoint=None
    ):
        """
        Args:
            img_channels (int, optional): Number of channels of the images [Default: 1]
            num_in_features (int, optional): Number of channels that are created from
                the input to feed to the first dense block [Default: 48]
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
            checkpoint (dict, optional): State dictionary to be loaded
        """
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(
            img_channels,
            num_in_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        num_features = num_in_features
        self.block1 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False
        )
        self.trans2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.multi_block = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=multi_block_depth,
            dropout_rate=dropout_rate,
        )
        num_features = num_features // 2
        self.block3 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        # Add self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=num_features,
            num_heads=8,  # Number of attention heads, needs to be modified
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)
        out_A = self.trans2_pool(out_A)
        out_A = self.block3(out_A)
        out_B = self.multi_block(out_before_trans2)

        # Apply self-attention
        out_A, _ = self.self_attention(out_A, out_A, out_A)

        return out_A, out_B


class CoverageAttention(nn.Module):
    """Coverage attention

    The coverage attention is a multi-layer perceptron, which takes encoded annotations
    and creates a context vector.
    """

    # input_size = C
    # output_size = q
    # attn_size = L = H * W
    def __init__(
        self,
        input_size,
        output_size,
        attn_size,
        kernel_size,
        padding=0,
        device=device,
    ):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the coverage
            attn_size (int): Length of the annotation vector
            kernel_size (int): Kernel size of the 1D convolutional layer
            padding (int, optional): Padding of the 1D convolutional layer [Default: 0]
            device (torch.device, optional): Device for the tensors
        """
        super(CoverageAttention, self).__init__()
        self.alpha = None
        self.conv = nn.Conv2d(1, output_size, kernel_size=kernel_size, padding=padding)
        self.U_a = nn.Parameter(torch.empty((n_prime, input_size)))
        self.U_f = nn.Parameter(torch.empty((n_prime, output_size)))
        self.nu_attn = nn.Parameter(torch.empty(n_prime))
        self.input_size = input_size
        self.output_size = output_size
        self.attn_size = attn_size
        self.device = device
        nn.init.xavier_normal_(self.U_a)
        nn.init.xavier_normal_(self.U_f)
        # Xavier requires at least a 2D tensor.
        nn.init.xavier_normal_(self.nu_attn.unsqueeze(0))

    def reset_alpha(self, batch_size):
        self.alpha = torch.zeros((batch_size, 1, self.attn_size), device=self.device)

    def forward(self, x, u_pred):
        batch_size = x.size(0)
        if self.alpha is None:
            self.reset_alpha(batch_size)
        # Change the dimensions to make it possible to apply a 2D convolution
        # From: (batch_size x L)
        # To: (batch_size x H x W)
        alpha_sum = self.alpha.sum(1).view(batch_size, x.size(2), x.size(3))
        conv_out = self.conv(alpha_sum.unsqueeze(1))
        # Change dimensions back
        # From: (batch_size x output_size x H x W)
        # To: (batch_size x output_size x L)
        conv_out = conv_out.view(batch_size, self.output_size, -1)
        # Change the dimensions
        # From: (batch_size x C x H x W)
        # To: (batch_size x C x L)
        a = x.view(batch_size, x.size(1), -1)
        u_a = torch.matmul(self.U_a, a)
        u_f = torch.matmul(self.U_f, conv_out)
        # u_pred is expanded from (batch_size x n_prime)
        # to (batch_size x n_prime x L) because there are L components to which
        # the same u_pred is added.
        u_pred_expanded = u_pred.unsqueeze(2).expand_as(u_a)
        tan_res = torch.tanh(u_pred_expanded + u_a + u_f)
        e_t = torch.matmul(self.nu_attn, tan_res)
        alpha_t = torch.softmax(e_t, dim=1)
        self.alpha = torch.cat((self.alpha, alpha_t.detach().unsqueeze(1)), dim=1)
        # alpha_t: (batch_size x L)
        # a: (batch_size x C x L) but need (C x batch_size x L) for
        # element-wise multiplication. So transpose them.
        cA_t_L = alpha_t * a.transpose(0, 1)
        # Transpose back
        return cA_t_L.transpose(0, 1).sum(2)


class Maxout(nn.Module):
    """
    Maxout makes pools from the last dimension and keeps only the maximum value from
    each pool.
    """

    def __init__(self, pool_size):
        """
        Args:
            pool_size (int): Number of elements per pool
        """
        super(Maxout, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        [*shape, last] = x.size()
        out = x.view(*shape, last // self.pool_size, self.pool_size)
        out, _ = out.max(-1)
        return out


class Decoder(nn.Module):
    """Decoder

    GRU based Decoder which attends to the low- and high-resolution annotations to
    create a LaTeX string.
    """

    def __init__(
        self,
        num_classes,
        low_res_shape,
        high_res_shape,
        hidden_size=256,
        embedding_dim=256,
        checkpoint=None,
        device=device,
    ):
        """
        Args:
            num_classes (int): Number of symbol classes
            low_res_shape ((int, int, int)): Shape of the low resolution annotations
                i.e. (C, W, H)
            high_res_shape ((int, int, int)): Shape of the high resolution annotations
                i.e. (C_prime, 2W, 2H)
            hidden_size (int, optional): Hidden size of the GRU [Default: 256]
            embedding_dim (int, optional): Dimension of the embedding [Default: 256]
            checkpoint (dict, optional): State dictionary to be loaded
            device (torch.device, optional): Device for the tensors
        """
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.low_res_shape = low_res_shape
        self.high_res_shape = high_res_shape
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.device = device

        self.low_res_embedding = nn.Embedding(
            num_classes, embedding_dim, padding_idx=0
        )
        self.high_res_embedding = nn.Embedding(
            num_classes, embedding_dim, padding_idx=0
        )

        # Add attention layers
        self.low_res_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,  # Number of attention heads, needs to be modified
        )
        self.high_res_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,  # Number of attention heads, needs to be modified
        )

        self.gru = nn.GRU(
            embedding_dim + hidden_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))

    def reset(self, batch_size):
        self.coverage_attn_low.reset_alpha(batch_size)
        self.coverage_attn_high.reset_alpha(batch_size)

    # Unsqueeze and squeeze are used to add and remove the seq_len dimension,
    # which is always 1 since only the previous symbol is provided, not a sequence.
    # The inputs that are multiplied by the weights are transposed to get
    # (m x batch_size) instead of (batch_size x m). The result of the
    # multiplication is tranposed back.
    def forward(self, low_res_annot, high_res_annot, target=None, teacher_force_ratio=0.5):
        batch_size = low_res_annot.size(0)
        seq_len = low_res_annot.size(1)

        low_res_embedded = self.low_res_embedding(low_res_annot)
        high_res_embedded = self.high_res_embedding(high_res_annot)

        hidden = torch.zeros(2, batch_size, self.hidden_size).to(self.device)

        outputs = []
        for t in range(seq_len):
            low_res_input = low_res_embedded[:, t, :].unsqueeze(1)
            high_res_input = high_res_embedded[:, t, :].unsqueeze(1)

            # Apply attention
            low_res_output, _ = self.low_res_attention(
                low_res_input.transpose(0, 1), low_res_annot.transpose(1, 2), low_res_annot.transpose(1, 2)
            )
            high_res_output, _ = self.high_res_attention(
                high_res_input.transpose(0, 1), high_res_annot.transpose(1, 2), high_res_annot.transpose(1, 2)
            )

            rnn_input = torch.cat([low_res_output.squeeze(1), high_res_output.squeeze(1)], dim=1)

            output, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
            output = torch.squeeze(output, 1)
            output = self.fc(output)

            outputs.append(output)

            if target is not None and random.random() < teacher_force_ratio:
                output = target[:, t]

            low_res_embedded = self.low_res_embedding(output).unsqueeze(1)
            high_res_embedded = self.high_res_embedding(output).unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)

        return outputs
