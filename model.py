import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

from constants import SAMPLE_RATE, N_MELS, N_FFT, F_MAX, F_MIN, HOP_SIZE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, normalized=False)
    
    def forward(self, audio):
        batch_size = audio.shape[0]
        
        # alignment correction to match with pianoroll
        # pretty_midi.get_piano_roll use ceil, but torchaudio.transforms.melspectrogram uses
        # round when they convert the input into frames.
        padded_audio = nn.functional.pad(audio, (N_FFT // 2, 0), 'constant')
        mel = self.melspectrogram(audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        mel = th.log(th.clamp(mel, min=1e-9))
        return mel



class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size * 1 channel * frames * input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        #print(x.shape)
        x = self.cnn(x)
        #print(x.shape)
        x = x.transpose(1, 2).flatten(-2)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x


class Transcriber(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc = nn.Linear(fc_unit, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)

        x = self.frame_conv_stack(mel)  # (B x T x C)
        #print(x.shape)
        frame_out = self.frame_fc(x)
        #print(frame_out.shape)

        x = self.onset_conv_stack(mel)  # (B x T x C)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out

class Transcriber_RNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        
        
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_lstm = nn.LSTM(input_size = 229, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)
        
        self.onset_lstm = nn.LSTM(input_size = 229, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)

        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        out1, _ = self.frame_lstm(mel, (frame_h0,frame_c0))
        #print(out1.shape)
        frame_out = self.frame_fc(out1)
        #print(frame_out.shape)
        
        out2, _ = self.onset_lstm(mel, (onset_h0,onset_c0))
        onset_out = self.onset_fc(out2)
        
        return frame_out, onset_out


class Transcriber_CRNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()      
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1, _ = self.frame_lstm(out1, (frame_h0,frame_c0))
        #print(out1.shape)
        frame_out = self.frame_fc(out1)
        #print(frame_out.shape)
        
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        onset_out = self.onset_fc(out2) 
        
        return frame_out, onset_out
    
        
class Transcriber_ONF(nn.Module):  # original design with killing gradient
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_lstm = nn.LSTM(input_size = 88+88, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc2 = nn.Linear(88*2, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)
        
        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        # onset
        #print(mel.shape)
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        #print(out2.shape)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        #print(out2.shape)
        onset_out = self.onset_fc(out2)
        #print(onset_out.shape)

        # frame
        #print(mel.shape)
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)
        
        ### copy onset_out by killing its gradient
        onset_out_copy = onset_out.clone() # copy it on new variable
        onset_out_copy = onset_out_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,onset_out_copy), 2) # concatenate it
        
        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        #print(out1.shape)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)


        return frame_out, onset_out
    
    
    
class Transcriber_ONF2(nn.Module):  # original design without killing gradient
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_lstm = nn.LSTM(input_size = 88+88, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc2 = nn.Linear(88*2, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)
        
        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        
        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        # onset
        #print(mel.shape)
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        #print(out2.shape)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        #print(out2.shape)
        onset_out = self.onset_fc(out2)
        #print(onset_out.shape)

        # frame
        #print(mel.shape)
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)
        out1 = th.cat((out1,onset_out), 2) # concatenate it
        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        #print(out1.shape)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)


        return frame_out, onset_out
    
    
class Transcriber_ONF3(nn.Module):   # ONF without first FC layer and with killing gradient
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_lstm = nn.LSTM(input_size =fc_unit+88,hidden_size = 88,num_layers = 2,batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)

        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)
        
        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        # onset
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        onset_out = self.onset_fc(out2)

        # frame
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        ### copy onset_out by killing its gradient
        onset_out_copy = onset_out.clone() # copy it on new variable
        onset_out_copy = onset_out_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,onset_out_copy), 2) # concatenate it
        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        frame_out = self.frame_fc(out1)



        return frame_out, onset_out
    
    
    
class Transcriber_ONF4(nn.Module):    # ONF without first FC layer and without killing gradient
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_lstm = nn.LSTM(input_size =fc_unit+88,hidden_size = 88,num_layers = 2,batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)

        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)
        
        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        # onset
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        onset_out = self.onset_fc(out2)

        # frame
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        out1 = th.cat((out1,onset_out), 2) # concatenate it
        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        frame_out = self.frame_fc(out1)


        return frame_out, onset_out
    
    
class Transcriber_ONF5(nn.Module):  # original design without killing gradient and with lstm connection
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_lstm = nn.LSTM(input_size = 88+88*2, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc2 = nn.Linear(88*2, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)
        
        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        # onset
        #print(mel.shape)
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        #print(out2.shape)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        #print(out2.shape)
        onset_out = self.onset_fc(out2)
        #print(onset_out.shape)

        # frame
        #print(mel.shape)
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)
        
        ### copy onset_out by killing its gradient
        out2_copy = out2.clone() # copy it on new variable
        out2_copy = out2.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,out2_copy), 2) # concatenate it
        
        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        #print(out1.shape)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)


        return frame_out, onset_out
    
    
    
###################################################################################    
###### Transformer implementation

## selfattention block
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = th.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = th.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = th.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
    
class Encoder_transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout):

        super(Encoder_transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
 
 #similar to ONF, LSTM block is changed to transformer encoder, killing gradient
class Transcriber_Transformer(nn.Module):   
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = 88*2, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda", 
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.frame_fc2 = nn.Linear(88*2, 88)
        
        
 
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc1 = nn.Linear(fc_unit, 88)
        self.onset_transformer = Encoder_transformer( 
                                   embed_size = 88, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda",
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.onset_fc2 = nn.Linear(88, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        
        out2 = self.onset_conv_stack(mel)
        #print(out2.shape)
        out2 = self.onset_fc1(out2)
        #print(out2.shape)
        out2 = self.onset_transformer(out2, None)
        #print(out2.shape)
        onset_out = self.onset_fc2(out2)
        #print(onset_out.shape)
        
        out1 = self.frame_conv_stack(mel)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)
        
        ### copy onset_out by killing its gradient
        onset_out_copy = onset_out.clone() # copy it on new variable
        onset_out_copy = onset_out_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,onset_out_copy), 2) # concatenate it
        #print(out1.shape)
        
        out1 = self.frame_transformer(out1, None)
        #print(out1.shape)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)
     
        return frame_out, onset_out

    
    
class Transcriber_Transformer2(nn.Module):   # just transfer block on mel
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = 229, 
                                   num_layers = 6, 
                                   heads = 1,
                                   device = "cuda", 
                                   forward_expansion = 2,
                                   dropout = 0) 
        self.frame_fc = nn.Linear(229, 88)
        
        

        self.onset_transformer = Encoder_transformer( 
                                   embed_size = 229, 
                                   num_layers = 6, 
                                   heads = 1,
                                   device = "cuda",
                                   forward_expansion = 2,
                                   dropout = 0) 
        self.onset_fc = nn.Linear(229, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)

        out1 = self.frame_transformer(mel, None)
        frame_out = self.frame_fc(out1)
        
        out2 = self.onset_transformer(mel, None)
        onset_out = self.onset_fc(out2)
        


     
        return frame_out, onset_out
    
    
class Transcriber_Transformer3(nn.Module):   # just transfer block on mel with connection between onset and frame transformer
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = 229*2, 
                                   num_layers = 6, 
                                   heads = 1,
                                   device = "cuda", 
                                   forward_expansion = 2,
                                   dropout = 0) 
        self.frame_fc = nn.Linear(229*2, 88)
        
        

        self.onset_transformer = Encoder_transformer( 
                                   embed_size = 229, 
                                   num_layers = 6, 
                                   heads = 1,
                                   device = "cuda",
                                   forward_expansion = 2,
                                   dropout = 0) 
        self.onset_fc = nn.Linear(229, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        
        out2 = self.onset_transformer(mel, None)  ## ---> this will be concatinated (transformer output to transfer input)
        onset_out = self.onset_fc(out2)
        
        
        ### copy onset_out by killing its gradient
        out2_copy = out2.clone() # copy it on new variable
        out2_copy = out2_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((mel,out2_copy), 2) # concatenate it
        out1 = self.frame_transformer(out1, None)
        frame_out = self.frame_fc(out1)

     
        return frame_out, onset_out
    
#similar to ONF, without firt FC, LSTM block is changed to transformer encoder, killing gradient
class Transcriber_Transformer4(nn.Module):    
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = fc_unit+88, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda", 
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.frame_fc = nn.Linear(fc_unit+88, 88)
        
        
 
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_transformer = Encoder_transformer( 
                                   embed_size = fc_unit, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda",
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.onset_fc = nn.Linear(fc_unit, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        
        out2 = self.onset_conv_stack(mel)
        #print(out2.shape)
        out2 = self.onset_transformer(out2, None)
        #print(out2.shape)
        onset_out = self.onset_fc(out2)
        #print(onset_out.shape)
        
        out1 = self.frame_conv_stack(mel)
        #print(out1.shape)
        
        ### copy onset_out by killing its gradient
        onset_out_copy = onset_out.clone() # copy it on new variable
        onset_out_copy = onset_out_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,onset_out_copy), 2) # concatenate it
        #print(out1.shape)
        
        out1 = self.frame_transformer(out1, None)
        #print(out1.shape)
        frame_out = self.frame_fc(out1)
        #print(frame_out.shape)
     
        return frame_out, onset_out     
    
    
    
    
#similar to ONF, without firt FC, LSTM block is changed to transformer encoder, killing gradient
class Transcriber_Transformer5(nn.Module):    
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = fc_unit*2, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda", 
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.frame_fc = nn.Linear(fc_unit*2, 88)
        
        
 
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_transformer = Encoder_transformer( 
                                   embed_size = fc_unit, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda",
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.onset_fc = nn.Linear(fc_unit, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        
        out2 = self.onset_conv_stack(mel)
        out2 = self.onset_transformer(out2, None)
        onset_out = self.onset_fc(out2)
        
        out1 = self.frame_conv_stack(mel)
        #print(out1.shape)
        
        ### copy onset_out by killing its gradient
        out2_copy = out2.clone() # copy it on new variable
        out2_copy = out2_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,out2_copy), 2) # concatenate it
        #print(out1.shape)
        
        out1 = self.frame_transformer(out1, None)
        #print(out1.shape)
        frame_out = self.frame_fc(out1)
        #print(frame_out.shape)
     
        return frame_out, onset_out    
    
    
    
class Transcriber_Transformer6(nn.Module):  # original design with killing gradient (transformer+lstm)
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_lstm = nn.LSTM(input_size = 88+88, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = 88*2, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda", 
                                   forward_expansion = 4,
                                   dropout = 0) 
        self.frame_fc2 = nn.Linear(88*2, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_transformer = Encoder_transformer( 
                                   embed_size = 88*2, 
                                   num_layers = 6, 
                                   heads = 8,
                                   device = "cuda", 
                                   forward_expansion = 4,
                                   dropout = 0) 
        
        self.onset_fc = nn.Linear(88*2, 88)
        
        
    def forward(self, audio):
        mel = self.melspectrogram(audio)
        

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        
        # onset
        #print(mel.shape)
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        #print(out2.shape)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        #print(out2.shape)
        out2 = self.onset_transformer(out2, None)
        onset_out = self.onset_fc(out2)
        #print(onset_out.shape)

        # frame
        #print(mel.shape)
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)
        
        ### copy onset_out by killing its gradient
        onset_out_copy = onset_out.clone() # copy it on new variable
        onset_out_copy = onset_out_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,onset_out_copy), 2) # concatenate it
        
        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        #print(out1.shape)
        out1 = self.frame_transformer(out1, None)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)


        return frame_out, onset_out
    
    
'''
class Transcriber_Transformer(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_transformer = Encoder_transformer( 
                                   embed_size = 88, 
                                   num_layers = 1, 
                                   heads = 4,
                                   device = "cuda", 
                                   forward_expansion = 2,
                                   dropout = 0) 
        self.frame_fc2 = nn.Linear(88, 88)
        
        
 
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc1 = nn.Linear(fc_unit, 88)
        self.onset_transformer = Encoder_transformer( 
                                   embed_size = 88, 
                                   num_layers = 1, 
                                   heads = 4,
                                   device = "cuda",
                                   forward_expansion = 2,
                                   dropout = 0) 
        self.onset_fc2 = nn.Linear(88, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)
        
        out1 = self.frame_conv_stack(mel)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)
        out1 = self.frame_transformer(out1, None)
        #print(out1.shape)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)
        
        
        out2 = self.onset_conv_stack(mel)
        out2 = self.onset_fc1(out2)
        out2 = self.onset_transformer(out2, None)
        onset_out = self.onset_fc2(out2)
        
     
        return frame_out, onset_out

'''