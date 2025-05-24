import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CustomEncoderCNN(nn.Module):
    """
    Custom CNN encoder based on the architecture description:
    ME Images (h×w×d) -> FCN output (H×W×D)
    Architecture: 4×conv3-32 -> maxpool -> 4×conv3-64 -> maxpool -> 4×conv3-128 -> maxpool -> FCN output
    """
    def __init__(self, enc_hidden_size=256, input_channels=3):
        super(CustomEncoderCNN, self).__init__()
        
        # First block: 4×conv3-32
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block: 4×conv3-64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third block: 4×conv3-128
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Additional conv block for further feature extraction
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final conv layer to match encoder hidden size
        self.conv_reduce = nn.Conv2d(256, enc_hidden_size, kernel_size=1)
        
    def forward(self, images):
        """
        Extract features from input images and reshape for attention mechanism
        images: [batch_size, channels, height, width]
        Returns: [batch_size, height*width, feature_size]
        """
        # Forward through conv blocks
        x = self.conv_block1(images)  # [B, 32, H, W]
        x = self.maxpool1(x)          # [B, 32, H/2, W/2]
        
        x = self.conv_block2(x)       # [B, 64, H/2, W/2]
        x = self.maxpool2(x)          # [B, 64, H/4, W/4]
        
        x = self.conv_block3(x)       # [B, 128, H/4, W/4]
        x = self.maxpool3(x)          # [B, 128, H/8, W/8]
        
        x = self.conv_block4(x)       # [B, 256, H/8, W/8]
        x = self.maxpool4(x)          # [B, 256, H/16, W/16]
        
        features = self.conv_reduce(x)  # [B, enc_hidden_size, H/16, W/16]
        
        # Reshape for attention mechanism
        batch_size = features.size(0)
        feature_size = features.size(1)
        height, width = features.size(2), features.size(3)
        
        # Permute and reshape: [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size, height * width, feature_size)
        
        return features


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism with coverage
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
        """
        num_pixels = encoder_out.size(1)
        encoder_att = self.encoder_att(encoder_out)
        decoder_att = self.decoder_att(decoder_hidden)
        decoder_att = decoder_att.unsqueeze(1)

        if coverage is None:
            coverage = torch.zeros(encoder_out.size(0), num_pixels, 1).to(encoder_out.device)
        coverage_att = self.coverage_att(coverage)

        att = torch.tanh(encoder_att + decoder_att + coverage_att)
        att = self.full_att(att).squeeze(2)
        alpha = F.softmax(att, dim=1)
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
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
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        """
        Initialize hidden state and cell state for the LSTM
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass for training
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        coverage = torch.zeros(batch_size, num_pixels, 1).to(encoder_out.device)
        coverage_seq = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            context_vector, alpha, coverage = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
                coverage[:batch_size_t] if t > 0 else None
            )
            coverage_seq[:batch_size_t, t, :] = coverage.squeeze(2)
            
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            context_vector = gate * context_vector
            
            h, c = self.lstm_cell(
                torch.cat([embeddings[:batch_size_t, t, :], context_vector], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def generate_caption(self, encoder_out, max_length=150, start_token=1, end_token=2):
        """
        Generate captions (LaTeX sequences)
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "batch prediction is not supported"
        
        predictions = []
        alphas = []
        h, c = self.init_hidden_state(encoder_out)
        prev_word = torch.LongTensor([start_token]).to(encoder_out.device)
        coverage = None

        for i in range(max_length):
            embeddings = self.embedding(prev_word)
            context_vector, alpha, coverage = self.attention(encoder_out, h, coverage)
            
            gate = torch.sigmoid(self.f_beta(h))
            context_vector = gate * context_vector
            
            h, c = self.lstm_cell(
                torch.cat([embeddings, context_vector], dim=1),
                (h, c)
            )
            
            preds = self.fc(h)
            _, next_word = torch.max(preds, dim=1)
            predictions.append(next_word.item())
            alphas.append(alpha)
            prev_word = next_word
            
            if next_word.item() == end_token:
                break

        return predictions, alphas


class WAP(nn.Module):
    """
    Watch, Attend and Parse model with custom CNN encoder
    """
    def __init__(self, vocab_size, embed_size=256, encoder_dim=256, decoder_dim=512, attention_dim=256, dropout=0.5):
        super(WAP, self).__init__()
        self.encoder = CustomEncoderCNN(enc_hidden_size=encoder_dim)
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
        """
        encoder_out = self.encoder(images)
        predictions, alphas, coverage_seq, decode_lengths, sort_ind = self.decoder(
            encoder_out, encoded_captions, caption_lengths
        )
        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def recognize(self, image, max_length=150, start_token=1, end_token=2):
        """
        Recognize handwritten mathematical expression and output LaTeX sequence
        """
        batch_size = image.size(0)
        assert batch_size == 1, "batch prediction is not supported"
        
        encoder_out = self.encoder(image)
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
    my_model = CustomEncoderCNN(256)
    example_image = torch.randn(1, 3, 224, 224)
    output = my_model.forward(example_image)
    print(output)
    print(output.shape)


if __name__ == '__main__':
    main()
