import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN-based encoder for extracting visual features from handwritten math expressions
    """

    def __init__(self, enc_hidden_size=256):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet18(weights=None)
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # for param in resnet.parameters():
        #     param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # conv layers to reduce feature map dimensions, from 512 -> enc_hidden_size
        self.conv_reduce = nn.Conv2d(512, enc_hidden_size, kernel_size=1)

    def forward(self, images):
        """
        Extract features from input images; then reshape for attention mechanism

        images: [batch_size, channels, height, width]
        """
        # [batch_size, 512, h/32, w/32]
        features = self.resnet(images)

        # [batch_size, hidden_size, h/32, w/32]
        features = self.conv_reduce(features)

        batch_size = features.size(0)
        feature_size = features.size(1)
        height, width = features.size(2), features.size(3)

        # [batch_size, height*width, feature_size] or [batch_size, h*w/1024, hidden_size]
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size, height*width, feature_size)

        return features


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism with coverage

    encoder_att: projects encoder output to the attention dimension

    decoder_att: projects decoder hidden state to the attention dimension

    coverage_att: projects the coverage vector to the attention dimension

    full_att: computes a scalar attention score from the combined feature
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.coverage_att = nn.Linear(1, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden, coverage=None):
        """
        Calculate context vector for the current time step

        encoder_out: [batch_size, num_pixels, encoder_dim]

        decoder_hidden: [batch_size, decoder_dim]

        coverage: [batch_size, num_pixels, 1]
        """

        num_pixels = encoder_out.size(1)

        # project inputs to attention space

        # [batch_size, num_pixels, attention_dim]
        encoder_att = self.encoder_att(encoder_out)

        # [batch_size, attention_dim]
        decoder_att = self.decoder_att(decoder_hidden)
        # [batch_size, 1, attention_dim]
        decoder_att = decoder_att.unsqueeze(1)

        if coverage is None:
            # first time step, no coverage
            # [batch_size, num_pixels, 1]
            coverage = torch.zeros(encoder_out.size(0), num_pixels, 1).to(encoder_out.device)
        # [batch_size, num_pixels, attention_dim]
        coverage_att = self.coverage_att(coverage)


        # add and apply nonlinearity in Bahdanau-style
        # [batch_size, num_pixels, attention_dim]
        att = torch.tanh(encoder_att + decoder_att + coverage_att)

        # compute scalar attention scores
        # [batch_size, num_pixels, 1] -> [batch_size, num_pixels]
        att = self.full_att(att).squeeze(2)  
        # [batch_size, num_pixels]
        alpha = F.softmax(att, dim=1)
        # print('alpha:', alpha.shape)

        # calculate attention-weighted context vector by weighting
        # each encoder output feature with the corresponding attention weight
        # [batch_size, encoder_dim]
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        # print('context_vec:', context_vector.shape)
        
        # update coverage vector
        # [batch_size, num_pixels, 1]
        coverage = coverage + alpha.unsqueeze(2)

        return context_vector, alpha, coverage


class DecoderRNN(nn.Module):
    """
    LSTM-based decoder with attention
    """

    def __init__(self, vocab_size, embed_size, encoder_dim, decoder_dim, attention_dim, dropout=0.5):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(
            encoder_dim, decoder_dim, attention_dim)

        # Create LSTMCell instead of LSTM to have more control over each time step
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        """
        Initialize hidden state and cell state for the LSTM

        encoder_out: [batch_size, num_pixels, encoder_dim]
        """
        
        # initialize hidden state and cell state by mean across pixels
        # [batch_size, encoder_dim]
        mean_encoder_out = encoder_out.mean(dim=1)
        # [batch_size, decoder_dim]
        h = self.init_h(mean_encoder_out)
        # [batch_size, decoder_dim]
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass for training

        encoder_out: [batch_size, num_pixels, encoder_dim]

        encoded_captions: [batch_size, max_caption_length]
        
        caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # sort by caption length
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # embedding
        # [batch_size, max_caption_length, embed_size]
        embeddings = self.embedding(encoded_captions)

        # initialize LSTM state
        # [batch_size, decoder_dim]
        h, c = self.init_hidden_state(encoder_out)

        # we won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # so, decoding lengths are caption lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word prediction and attention alpha values
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # initialize coverage vector
        coverage = torch.zeros(batch_size, num_pixels, 1).to(encoder_out.device)
        coverage_seq = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # for each time step
        for t in range(max(decode_lengths)):
            # focus on certain batch elements based on decode_lengths
            batch_size_t = sum([l > t for l in decode_lengths])

            # compute attention
            context_vector, alpha, coverage = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
                coverage[:batch_size_t] if t > 0 else None
            )

            coverage_seq[:batch_size_t, t, :] = coverage.squeeze(2)

            # gate
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            context_vector = gate * context_vector

            # LSTM step
            h, c = self.lstm_cell(
                torch.cat([embeddings[:batch_size_t, t, :], context_vector], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            # Generate prediction
            # [batch_size_t, vocab_size]
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def generate_caption(self, encoder_out, max_length=150, start_token=1, end_token=2):
        """
        Generate captions (LaTeX sequences)

        note: prediction will not have start_token

        encoder_out: [1, num_pixels, encoder_dim]
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "batch prediction is not supported"

        # initialize tensors to hold predictions and attention alphas
        predictions = []
        alphas = []

        # initialize hidden states
        h, c = self.init_hidden_state(encoder_out)

        # start with start token
        prev_word = torch.LongTensor([start_token]).to(encoder_out.device)

        # initialize coverage vector
        coverage = None

        for i in range(max_length):
            # embed the previous word
            # [1, embed_size]
            embeddings = self.embedding(prev_word)

            # attend
            context_vector, alpha, coverage = self.attention(
                encoder_out,
                h,
                coverage
            )

            # gate
            gate = torch.sigmoid(self.f_beta(h))
            context_vector = gate * context_vector

            # LSTM step
            h, c = self.lstm_cell(
                torch.cat([embeddings, context_vector], dim=1),
                (h, c)
            )

            # generate prediction
            # [1, vocab_size]
            preds = self.fc(h)

            # find next word: using max prob.
            _, next_word = torch.max(preds, dim=1)

            # add predictions and attentions
            predictions.append(next_word.item())
            alphas.append(alpha)

            # update previous word
            prev_word = next_word

            # break if end token is predicted
            if next_word.item() == end_token:
                break

        return predictions, alphas


class WAP(nn.Module):
    """
    Watch, Attend and Parse model for handwritten mathematical expression recognition
    """

    def __init__(self, vocab_size, embed_size=256, encoder_dim=256, decoder_dim=512, attention_dim=256, dropout=0.5):
        super(WAP, self).__init__()

        self.encoder = EncoderCNN(enc_hidden_size=encoder_dim)
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_size=embed_size,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

    def forward(self, images, encoded_captions, caption_lengths):
        """
        Forward pass

        images: [batch_size, channels, height, width]

        encoded_captions: [batch_size, max_caption_length]

        caption_lengths: [batch_size, 1]
        """
        # encode images
        # [batch_size, num_pixels, encoder_dim]
        encoder_out = self.encoder(images)

        # decode captions
        predictions, alphas, coverage_seq, decode_lengths, sort_ind = self.decoder(
            encoder_out, encoded_captions, caption_lengths
        )

        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def recognize(self, image, max_length=150, start_token=1, end_token=2):
        """
        Recognize handwritten mathematical expression and output LaTeX sequence

        image: [1, channels, height, width]
        """
        # encode image
        batch_size = image.size(0)
        assert batch_size == 1, "batch prediction is not supported"
        encoder_out = self.encoder(image)  # [1, num_pixels, encoder_dim]

        # generate caption
        predictions, alphas = self.decoder.generate_caption(
            encoder_out,
            max_length=max_length,
            start_token=start_token,
            end_token=end_token
        )

        return predictions, alphas

# def main():

#     vocab_size = 10
#     model = WAP(vocab_size=vocab_size)

#     example_image = torch.randn(1, 3, 69, 420)  # Example image
#     predictions, alphas = model.recognize(example_image)

#     print(f'Predicted LaTeX sequence: {predictions}')
#     print(len(predictions))


def main():
    # testing encoder
    my_model = EncoderCNN(256)
    example_image = torch.randn(1, 3, 224, 224)
    output = my_model.forward(example_image)
    print(output)
    print(output.shape)


if __name__ == '__main__':
    main()
