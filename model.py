import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class AutoModelforKlueDp(nn.Module):
    def __init__(self, config, args):
        super(AutoModelforKlueDp, self).__init__()
        hparams = args

        self.model = AutoModel.from_config(config)

        self.hidden_size = hparams.hidden_size
        self.input_size = self.model.config.hidden_size
        self.arc_space = hparams.arc_space
        self.type_space = hparams.type_space

        self.n_pos_labels = len(utils.get_pos_labels())
        # self.n_type_labels = len(utils.get_dp_labels())
        self.n_type_labels = (
            63  # FIXME : Among all 63 types, only some of them (38) exist in klue-dp
        )
        if args.no_pos:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(self.n_pos_labels + 1, hparams.pos_dim)# 차원 추가

        enc_dim = self.input_size * 2  # concatenate start and end subword
        if self.pos_embedding is not None:
            enc_dim += hparams.pos_dim

        self.encoder = nn.LSTM(
            enc_dim,
            self.hidden_size,
            hparams.encoder_layers,
            batch_first=True,
            dropout=0.0 if hparams.encoder_layers == 1 else 0.33,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            hparams.decoder_layers,
            batch_first=True,
            dropout=0.0 if hparams.encoder_layers == 1 else 0.33,
        )

        self.dropout = nn.Dropout2d(p=0.33)

        self.src_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.hx_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.arc_c = nn.Linear(self.hidden_size * 2, self.arc_space)
        self.type_c = nn.Linear(self.hidden_size * 2, self.type_space)
        self.arc_h = nn.Linear(self.hidden_size, self.arc_space)
        self.type_h = nn.Linear(self.hidden_size, self.type_space)

        self.attention = BiAttention(self.arc_space, self.arc_space, 1)
        self.bilinear = BiLinear(self.type_space, self.type_space, self.n_type_labels)

    def forward(
        self,
        bpe_head_mask,
        bpe_tail_mask,
        pos_ids,
        head_ids,
        max_word_length,
        mask_e,
        mask_d,
        batch_index,
        input_ids,
        attention_mask,
    ):
        # pretrained language model, 
        # In our approach, we use a pretrained language model (to be fine-tuned) to extract subword representations
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.model(**inputs)
        outputs = outputs[0]#torch.Size([8, 128, 1024]), outputs.last_hidden_state
        # resizing outputs for top encoder-decoder layers
        # concatenate the first and last subword token representations of each word, to form word vector representations. 
        outputs, sent_len = utils.resize_outputs(##torch.Size([8, 26, 2048]), len(sent_len) = 8
            outputs, bpe_head_mask, bpe_tail_mask, max_word_length
        )
        # use pos features as inputs if available
        # Each of these word representations is optionally concatenated with the part-of-speech embedding.
        if self.pos_embedding is not None:
            pos_outputs = self.pos_embedding(pos_ids)
            pos_outputs = self.dropout(pos_outputs)#torch.Size([8, 26, 256])
            outputs = torch.cat([outputs, pos_outputs], dim=2)#torch.Size([8, 26, 2304])

        # encoder, we use biaffine attention [33] to predict HEAD
        packed_outputs = pack_padded_sequence(# 압축?
            outputs, sent_len, batch_first=True, enforce_sorted=False
        )#[torch.Size([130, 2304]), torch.Size([26]), torch.Size([8]), torch.Size([8])]
        encoder_outputs, hn = self.encoder(packed_outputs)#[torch.Size([130, 1536]), torch.Size([26]), torch.Size([8]), torch.Size([8])], [torch.Size([2, 8, 768]), torch.Size([2, 8, 768])],  torch.Size([2, 8, 768])
        encoder_outputs, outputs_len = pad_packed_sequence(# torch.Size([8, 26, 1536]), len(outputs_len) == 8
            encoder_outputs, batch_first=True# 해제? https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
        )
        encoder_outputs = self.dropout(encoder_outputs.transpose(1, 2)).transpose(
            1, 2
        )  # apply dropout for last layer, #  torch.Size([8, 26, 1536]) << torch.Size([8, 1536, 26]) <<  torch.Size([8, 26, 1536])
        hn = self._transform_decoder_init_state(hn)# [torch.Size([1, 8, 768]), torch.Size([1, 8, 768])]

        # decoder, and bilinear attention [64] to predict arc type (DEPREL) for each word
        src_encoding = F.elu(self.src_dense(encoder_outputs[:, 1:]))# torch.Size([8, 25, 768]) << torch.Size([8, 25, 768]) << torch.Size([8, 25, 1536])
        sent_len = [i - 1 for i in sent_len]# len(sent_len)==8
        packed_outputs = pack_padded_sequence(# [torch.Size([122, 768]), torch.Size([25]), torch.Size([8]), torch.Size([8])]
            src_encoding, sent_len, batch_first=True, enforce_sorted=False
        )
        decoder_outputs, _ = self.decoder(packed_outputs, hn)# [torch.Size([122, 768]), torch.Size([25]), torch.Size([8]), torch.Size([8])], [torch.Size([1, 8, 768]), torch.Size([1, 8, 768])]
        decoder_outputs, outputs_len = pad_packed_sequence(#torch.Size([8, 25, 768]), len(outputs_len) == 8
            decoder_outputs, batch_first=True
        )
        decoder_outputs = self.dropout(decoder_outputs.transpose(1, 2)).transpose(
            1, 2# torch.Size([8, 25, 768]) << torch.Size([8, 768, 25]) << torch.Size([8, 25, 768])
        )  # apply dropout for last layer

        # compute output for arc and type
        arc_c = F.elu(self.arc_c(encoder_outputs))# torch.Size([8, 26, 512]) << torch.Size([8, 26, 1536])
        type_c = F.elu(self.type_c(encoder_outputs))# torch.Size([8, 26, 256]) << torch.Size([8, 26, 1536])

        arc_h = F.elu(self.arc_h(decoder_outputs))# torch.Size([8, 25, 512]), torch.Size([8, 25, 768])
        type_h = F.elu(self.type_h(decoder_outputs))# torch.Size([8, 25, 256]), torch.Size([8, 25, 768])

        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(
            dim=1# torch.Size([8, 25, 26])
        )
        type_c = type_c[batch_index, head_ids.data.t()].transpose(0, 1).contiguous()# torch.Size([8, 25, 256]) << torch.Size([8, 26, 256])
        out_type = self.bilinear(type_h, type_c)# torch.Size([8, 25, 63])

        return out_arc, out_type# torch.Size([8, 25, 26]), torch.Size([8, 25, 63])

    def _transform_decoder_init_state(self, hn):#[torch.Size([2, 8, 768]), torch.Size([2, 8, 768])]
        hn, cn = hn# torch.Size([2, 8, 768]), torch.Size([2, 8, 768])
        cn = cn[-2:]  # take the last layer, torch.Size([2, 8, 768])
        _, batch_size, hidden_size = cn.size()
        cn = cn.transpose(0, 1).contiguous()# torch.Size([8, 2, 768])
        cn = cn.view(batch_size, 1, 2 * hidden_size).transpose(0, 1)#torch.Size([1, 8, 1536]) << torch.Size([8, 1, 1536])
        cn = self.hx_dense(cn)#torch.Size([1, 8, 768])
        if self.decoder.num_layers > 1:
            cn = torch.cat(
                [
                    cn,
                    torch.autograd.Variable(
                        cn.data.new(
                            self.decoder.num_layers - 1, batch_size, hidden_size
                        ).zero_()
                    ),
                ],
                dim=0,
            )
        hn = torch.tanh(cn)#torch.Size([1, 8, 768])
        hn = (hn, cn)#(torch.Size([1, 8, 768]), torch.Size([1, 8, 768]))
        return hn# len() == 2

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        # inference-speccific args : model
        parser.add_argument(
            "--encoder_layers", default=1, type=int, help="Number of layers of encoder"
        )
        parser.add_argument(
            "--decoder_layers", default=1, type=int, help="Number of layers of decoder"
        )
        parser.add_argument(
            "--hidden_size",
            default=768,
            type=int,
            help="Number of hidden units in LSTM",
        )
        parser.add_argument(
            "--arc_space", default=512, type=int, help="Dimension of tag space"
        )
        parser.add_argument(
            "--type_space", default=256, type=int, help="Dimension of tag space"
        )
        parser.add_argument(
            "--no_pos",
            default = False,# 20211002
            action='store_true',
            help="Use POS as input features in head layers",
        )
        parser.add_argument(
            "--pos_dim", default=256, type=int, help="Dimension of pos embedding"
        )
        args = parser.parse_args()
        if not args.no_pos and args.pos_dim <= 0:
            parser.error(
                "--pos_dim should be a positive integer when --no_pos is False."
            )
        return parser


class BiAttention(nn.Module):
    def __init__(
        self,
        input_size_encoder,
        input_size_decoder,
        num_labels,
        biaffine=True,
        **kwargs
    ):
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(
                torch.Tensor(
                    self.num_labels, self.input_size_decoder, self.input_size_encoder
                )
            )
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):# torch.Size([8, 25, 512]), torch.Size([8, 26, 512]), torch.Size([8, 25]), torch.Size([8, 26])
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()# 8, 25, 512 torch.Size([8, 25, 512])
        _, length_encoder, _ = input_e.size()# 8, 26, 512 <<  torch.Size([8, 26, 512])

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)# torch.Size([8, 1, 25, 1]) << torch.Size([8, 1, 25]) << torch.Size([1, 512]) * torch.Size([8, 1, 25])
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)# torch.Size([8, 1, 1, 26]) << torch.Size([1, 512]) * torch.Size([8, 512, 26]) 

        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)# torch.Size([8, 1, 25, 512]) << torch.Size([8, 1, 25, 512]), torch.Size([1, 512, 512])
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))# torch.Size([8, 1, 25, 26]) << torch.Size([8, 1, 25, 512]) << torch.Size([8, 1, 512, 26])
            output = output + out_d + out_e + self.b# torch.Size([8, 1, 25, 26]) <<  torch.Size([8, 1, 25, 26]) + torch.Size([8, 1, 25, 1]) + torch.Size([8, 1, 1, 26]) + torch.Size([1, 1, 1])
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = (
                output # torch.Size([8, 1, 25, 26]) * torch.Size([8, 1, 25, 1]) * torch.Size([8, 1, 1, 26])
                * mask_d.unsqueeze(1).unsqueeze(3)
                * mask_e.unsqueeze(1).unsqueeze(2)
            )

        return output# torch.Size([8, 1, 25, 26])


class BiLinear(nn.Module):
    def __init__(self, left_features, right_features, out_features):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(
            torch.Tensor(self.out_features, self.left_features, self.right_features)
        )
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):# torch.Size([8, 25, 256]),  torch.Size([8, 25, 256])
        left_size = input_left.size()# torch.Size([8, 25, 256])
        right_size = input_right.size()# torch.Size([8, 25, 256])
        assert (
            left_size[:-1] == right_size[:-1]
        ), "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))# 200 = 8*25

        input_left = input_left.reshape(batch, self.left_features)# torch.Size([200, 256]) << torch.Size([8, 25, 256])
        input_right = input_right.reshape(batch, self.right_features)# torch.Size([200, 256]) << torch.Size([8, 25, 256])

        output = F.bilinear(input_left, input_right, self.U, self.bias)# torch.Size([200, 63]) << torch.Size([200, 256]), torch.Size([200, 256]), torch.Size([63, 256, 256]) , torch.Size([63])
        output = (# torch.Size([200, 63])
            output# torch.Size([200, 63])
            + F.linear(input_left, self.W_l, None)# torch.Size([200, 63])
            + F.linear(input_right, self.W_r, None)# torch.Size([200, 63])
        )
        return output.view(left_size[:-1] + (self.out_features,))# torch.Size([8, 25, 63])
