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

        self.hidden_size = hparams.hidden_size # 768
        self.input_size = self.model.config.hidden_size# 1024
        self.arc_space = hparams.arc_space# 512
        self.type_space = hparams.type_space# 256

        self.n_pos_labels = len(utils.get_pos_labels())#  0,1,2 ... 44 {label: i for i, label in enumerate(get_pos_labels())}
        # self.n_type_labels = len(utils.get_dp_labels())
        self.n_type_labels = (
            # 63  # FIXME : Among all 63 types, only some of them (38) exist in klue-dp
            # DEPREL 0,1,2,...37 << {label: i for i, label in enumerate(get_dp_labels())}
            len(utils.get_dp_labels()) 
        )
        if args.no_pos:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(self.n_pos_labels + 1, hparams.pos_dim)# (45 + 1, 256) self.n_pos_labels + 1, data 처리할때도 열을 하나 더 추가 하는데 무슨의미인지 모름, 차원 추가

        enc_dim = self.input_size * 2  # concatenate start and end subword, 2048 = 1024 *2  
        if self.pos_embedding is not None:
            enc_dim += hparams.pos_dim# 2304 = 2048 + 256

        self.encoder = nn.LSTM(
            enc_dim,# 2304
            self.hidden_size,# 768
            hparams.encoder_layers,# 1
            batch_first=True,
            dropout=0.0 if hparams.encoder_layers == 1 else 0.33,
            bidirectional=True,# self.hidden_size * 2 반환
        )
        self.decoder = nn.LSTM(
            self.hidden_size,# 768
            self.hidden_size,# 768
            hparams.decoder_layers,# 1
            batch_first=True,
            dropout=0.0 if hparams.encoder_layers == 1 else 0.33,# encoder_layers? decoder 층에서 encoder 사용  
        )

        self.dropout = nn.Dropout2d(p=0.33)

        self.src_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)# 1536 >> 768 encoder(BiLSTM)의 output(첫번째 값 제외[CLS]) 값을 Decoder의 입력값으로 mapping
        self.hx_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)# 1536 >> 768 encoder(BiLSTM)의 last layer + last cell state 값을 Decoder의 입력값으로 mapping

        self.arc_c = nn.Linear(self.hidden_size * 2, self.arc_space)# 1536 >> 512 encoder(BiLSTM) >> BiAttention
        self.type_c = nn.Linear(self.hidden_size * 2, self.type_space)# 1536 >> 256  encoder(BiLSTM) >> BiLinear
        self.arc_h = nn.Linear(self.hidden_size, self.arc_space)# 768 >> 512 decoder(LSTM) >> BiAttention
        self.type_h = nn.Linear(self.hidden_size, self.type_space)# 768 >> 256 decoder(LSTM) >> BiLinear

        self.attention = BiAttention(self.arc_space, self.arc_space, 1)# (512, 512, 1),  we use biaffine attention [33] to predict HEAD
        self.bilinear = BiLinear(self.type_space, self.type_space, self.n_type_labels)# (256, 256, 38), , and bilinear attention [64] to predict arc type(DEPREL) for each word

    def forward(
        self,
        bpe_head_mask,#(8, 128)
        bpe_tail_mask,#(8, 128)
        pos_ids,#(8, 22)
        head_ids,#(8, 21)
        max_word_length,#21
        mask_e,#(8, 22)
        mask_d,#(8, 21)
        batch_index,# tensor([0, 1, 2, 3, 4, 5, 6, 7])
        input_ids,#(8, 128)
        attention_mask,#(8, 128)
    ):
        # pretrained language model, 
        # In our approach, we use a pretrained language model (to be fine-tuned) to extract subword representations
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.model(**inputs)# 'roberta'
        outputs = outputs.last_hidden_state # outputs[0]#torch.Size([8, 128, 1024]), outputs.last_hidden_state
        # resizing outputs for top encoder-decoder layers
        # concatenate the first and last subword token representations o f each word, to form word vector representations. 
        outputs, sent_len = utils.resize_outputs(##torch.Size([8, (max_word_length+1), 2048]), len(sent_len) = 8
            outputs, bpe_head_mask, bpe_tail_mask # [torch.Size([8, 128, 1024]), torch.Size([8, 128]), torch.Size([8, 128])]
            , max_word_length#21
        )
        # use pos features as inputs if available
        # Each of these word representations is optionally concatenated with the part-of-speech embedding.
        if self.pos_embedding is not None:
            pos_outputs = self.pos_embedding(pos_ids)
            pos_outputs = self.dropout(pos_outputs)#torch.Size([8, (max_word_length+1), 256])
            outputs = torch.cat([outputs, pos_outputs], dim=2)#torch.Size([8, (max_word_length+1), 2304])

        # encoder
        packed_outputs = pack_padded_sequence(# 압축?
            outputs, sent_len, batch_first=True, enforce_sorted=False
        )#[([130, 2304]), ([max_word_length+1]), ([8]), ([8])]
        encoder_outputs, hn  = self.encoder(packed_outputs)#[torch.Size([130, 768*2]), torch.Size([26]), torch.Size([8]), torch.Size([8])], [torch.Size([2, 8, 768]), torch.Size([2, 8, 768])],  torch.Size([2, 8, 768])
        encoder_outputs, outputs_len = pad_packed_sequence(# torch.Size([8, (max_word_length+1), 768*2]), len(outputs_len) == 8
            encoder_outputs, batch_first=True# 해제? https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
        )# ?? hn 값이 output중 하나에 포함이 되어 있어야 할것 같은데 못찾음 encoder_outputs.permute(1,0,2)[-1] == hn.permute(1,0,2).reshape(8,-1)
        encoder_outputs = self.dropout(encoder_outputs.transpose(1, 2)).transpose( # apply dropout for last layer 
            1, 2
        ) # ([8, (max_word_length+1), 768*2]) << ([8, 768*2, (max_word_length+1)]) <<  ([8, (max_word_length+1), 768*2])
        hn = self._transform_decoder_init_state(hn)# cell_state값으로 생성 [([1, 8, 768]), ([1, 8, 768])]

        # decoder
        src_encoding = F.elu(self.src_dense(encoder_outputs[:, 1:]))# encoder_outputs[:, 1:]?? ([8, max_word_length, 768])
        sent_len = [i - 1 for i in sent_len]# len(sent_len)==8
        packed_outputs = pack_padded_sequence(# [([122, 768]), ([max_word_length]), ([8]), ([8])]
            src_encoding, sent_len, batch_first=True, enforce_sorted=False
        )
        decoder_outputs, _ = self.decoder(packed_outputs, hn)# encoder의 output과 cell_state(last layer + activation info), 을 전부 사용 [([122, 768]), ([max_word_length]), ([8]), ([8])], [([1, 8, 768]), ([1, 8, 768])]
        decoder_outputs, outputs_len = pad_packed_sequence(#([8, 25, 768]), len(outputs_len) == 8
            decoder_outputs, batch_first=True
        )
        decoder_outputs = self.dropout(decoder_outputs.transpose(1, 2)).transpose(
            1, 2# ([8, max_word_length, 768]) << ([8, 768, max_word_length]) << ([8, max_word_length, 768])
        )  # apply dropout for last layer
        # compute output for arc and type
        arc_c = F.elu(self.arc_c(encoder_outputs))# ([8, (max_word_length+1), 512]) << ([8, (max_word_length+1), 1536])
        type_c = F.elu(self.type_c(encoder_outputs))# ([8, (max_word_length+1), 256]) <<([8, (max_word_length+1), 1536])
        arc_h = F.elu(self.arc_h(decoder_outputs))# ([8, max_word_length, 512]), ([8, max_word_length, 768])
        type_h = F.elu(self.type_h(decoder_outputs))# ([8, max_word_length, 256]), ([8, max_word_length, 768])

        # we use biaffine attention [33] to predict HEAD
        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(# ([8, max_word_length, (max_word_length+1)])
            dim=1# [([8, max_word_length, 512]), ([8, (max_word_length+1), 512]), ([8, max_word_length]), ([8, (max_word_length+1)])]
        )
        type_c = type_c[batch_index, head_ids.data.t()].transpose(0, 1).contiguous()# ([8, max_word_length, 256]) << ([8, (max_word_length+1), 256])
        # bilinear attention [64] to predict arc type (DEPREL) for each word
        out_type = self.bilinear(type_h, type_c)# ([8, max_word_length, self.n_type_labels])

        return out_arc, out_type# ([8, max_word_length, (max_word_length+1)]), ([8, max_word_length, self.n_type_labels])

    def _transform_decoder_init_state(self, hn):#[torch.Size([2, 8, 768]), torch.Size([2, 8, 768])]
        hn, cn = hn#  final hidden state, final cell state for each element in the batch.
        cn = cn[-2:]  # take the last layer, 마지막 layer의 cell states 는 cn[-2:]에 있음<< cn = ([2*num_layers, 8, 768])
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
        self.input_size_encoder = input_size_encoder # 512 self.arc_space
        self.input_size_decoder = input_size_decoder # 512 self.arc_space
        self.num_labels = num_labels# 1 ?
        self.biaffine = biaffine# True
        # torch.nn.parameter.Parameter:  A kind of Tensor that is to be considered a module parameter. 
        # when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator.
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))# (1,512) encoder에서 전달받은 vector(512)를 vector(1)로 mapping
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))# (1,512) decoder에서 전달받은 vector(512)를 vector(1)로 mapping
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))# (1,1,1)
        if self.biaffine:
            self.U = Parameter(
                torch.Tensor(
                    self.num_labels, self.input_size_decoder, self.input_size_encoder# (1, 512, 512)
                )
            )
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self):# Fills the input Tensor
        nn.init.xavier_uniform_(self.W_e)# (1, 512)
        nn.init.xavier_uniform_(self.W_d)# (1, 512)
        nn.init.constant_(self.b, 0.0)# (1, 1, 1)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)# (1, 512, 512)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):# ([8, max_word_length, 512]), ([8, (max_word_length+1), 512]), ([8, max_word_length]), ([8, (max_word_length+1)])
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()# ([8, max_word_length, 512])
        _, length_encoder, _ = input_e.size()# ([8, (max_word_length+1), 512])

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)# ([8, 1, max_word_length, 1]) << ([8, 1, max_word_length]) << ([1, 512]) * ([8, 512, max_word_length])
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)# ([8, 1, 1, (max_word_length+1)]) << ([1, 512]) * ([8, 512, (max_word_length+1)]) 

        if self.biaffine:# 뭐지?
            output = torch.matmul(input_d.unsqueeze(1), self.U)# ([8, 1, max_word_length, 512]) << ([8, 1, max_word_length, 512]) * ([1, 512, 512])
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))# ([8, 1, max_word_length, (max_word_length+1)]) << ([8, 1, max_word_length, 512]) * ([8, 1, 512, (max_word_length+1)])
            output = output + out_d + out_e + self.b# ([8, 1, max_word_length, (max_word_length+1)]) <<  ([8, 1, max_word_length, (max_word_length+1)]) + ([8, 1, max_word_length, 1]) + ([8, 1, 1, (max_word_length+1)]) + ([1, 1, 1])
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = (# 
                output # ([8, 1, max_word_length, (max_word_length+1)]) * ([8, 1, max_word_length, 1]) *([8, 1, 1, (max_word_length+1)])
                * mask_d.unsqueeze(1).unsqueeze(3)# ([8, 1, max_word_length, 1])        << ([8, max_word_length]) 
                * mask_e.unsqueeze(1).unsqueeze(2)# ([8, 1, 1, (max_word_length+1)])    << ([8, max_word_length+1])
            )
        return output# ([8, 1, max_word_length, (max_word_length+1)])

class BiLinear(nn.Module):
    def __init__(self, left_features, right_features, out_features):
        super(BiLinear, self).__init__()
        self.left_features = left_features# 256
        self.right_features = right_features# 256
        self.out_features = out_features# n_type_labels

        self.U = Parameter(
            torch.Tensor(self.out_features, self.left_features, self.right_features)# (n_type_labels, 256, 256)
        )
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))# (n_type_labels, 256)
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))# (n_type_labels, 256) self.left_features?? 항상 같은것이라고 보는건가?
        self.bias = Parameter(torch.Tensor(out_features))# n_type_labels

        self.reset_parameters()

    def reset_parameters(self):# Fills the input Tensor
        nn.init.xavier_uniform_(self.W_l)# (n_type_labels, 256)
        nn.init.xavier_uniform_(self.W_r)# (n_type_labels, 256)
        nn.init.constant_(self.bias, 0.0)# n_type_labels
        nn.init.xavier_uniform_(self.U)# (n_type_labels, 256, 256)

    def forward(self, input_left, input_right):#  from decoder, encoder // ([8, max_word_length, 256]),  ([8, max_word_length, 256])
        left_size = input_left.size()# torch.Size([8, max_word_length, 256])
        right_size = input_right.size()# torch.Size([8, max_word_length, 256])

        assert (
            left_size[:-1] == right_size[:-1]
        ), "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))# 200  8*max_word_length

        input_left = input_left.reshape(batch, self.left_features)# ([ 200, 256]) << ([8, max_word_length, 256])
        input_right = input_right.reshape(batch, self.right_features)# ([200 , 256]) << ([8, max_word_length, 256])

        output = F.bilinear(input_left, input_right, self.U, self.bias)# ([200 , n_type_labels]) << ([200, 256]), ([200, 256]), ([n_type_labels, 256, 256]) , ([n_type_labels])
        output = (# torch.Size([200, n_type_labels])
            output# torch.Size([200, n_type_labels])
            + F.linear(input_left, self.W_l, None)#     torch.Size([200, n_type_labels]) << ([200, 256]), ([n_type_labels, 256])
            + F.linear(input_right, self.W_r, None)#    torch.Size([200, n_type_labels]) << ([200, 256]), ([n_type_labels, 256])
        )
        return output.view(left_size[:-1] + (self.out_features,))# torch.Size([8, max_word_length, n_type_labels])
