import os
import torch
import torch.nn as nn
from torchinfo import summary
import lightning.pytorch as L
import numpy as np
import gin

from data.grandstaff import NUM_CHANNELS, SPECTROGRAM_HEIGHT, SCORE_HEIGHT, SOT_TOKEN, EOT_TOKEN, PAD_TOKEN
from utils.metrics import compute_metrics
from model.encoder import Encoder
from model.decoder import Decoder
from model.encoding import PositionalEncoding2D
    
####################################################### DAN MODEL:

@gin.configurable
class DAN(L.LightningModule):
    def __init__(self, d_model, dim_ff, maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, out_dir, encoder_type="DAN"):
        super().__init__()
        self.encoder = Encoder(in_channels) # Dropout = 0.5
        
        self.adaptor = None
        
        if encoder_type != "DAN":
            self.adaptor = nn.Conv2d(8, 256, kernel_size=1, stride=1)

        self.decoder = Decoder(d_model, dim_ff, maxlen, out_categories) # Attention window = 100
        self.positional_2D = PositionalEncoding2D(d_model, maxh, maxw)

        self.padding_token = padding_token

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.eximgs = []
        self.expreds = []
        self.exgts = []

        self.valpredictions = []
        self.valgts = []

        self.w2i = w2i
        self.i2w = i2w
        self.maxlen = maxlen
        self.out_dir=out_dir

        self.save_hyperparameters()

    def forward(self, x, y_pred):
        encoder_output = self.encoder(x)
        if self.adaptor != None:
            encoder_output = self.adaptor(encoder_output)
        
        return self.forward_decoder(encoder_output, y_pred, cache=None)

    def forward_encoder(self, x):
        if self.adaptor != None:
            return self.adaptor(self.encoder(x))
        return self.encoder(x)
    
    def forward_decoder(self, encoder_output, last_preds, cache=None):
        b, c, h, w = encoder_output.size()
        reduced_size = [s.shape[:2] for s in encoder_output]
        ylens = [len(sample) for sample in last_preds]
        cache = cache

        try:
            pos_features = self.positional_2D(encoder_output)
        except:
            self.positional_2D = PositionalEncoding2D(c, h, 1024)
            pos_features = self.positional_2D(encoder_output)

        features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2, 0, 1).contiguous()
        enhanced_features = features
        enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1).contiguous()
        output, predictions, _, _, weights = self.decoder(features, enhanced_features, last_preds[:, :], reduced_size, 
                                                          [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                          start=0, cache=cache, keep_all_weights=True)
    
        return output, predictions, cache, weights
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.0001, amsgrad=False)

    def training_step(self, train_batch):
        x, di, y = train_batch
        #print(f'di={di.shape}, y={y.shape}')
        output, predictions, cache, weights = self.forward(x, di)
        loss = self.loss(predictions, y)
        self.log("loss", loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, _, y = val_batch
        encoder_output = self.forward_encoder(x)
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i[SOT_TOKEN]])).to(device).unsqueeze(0)
        cache = None
        for i in range(self.maxlen):
            output, predictions, cache, weights = self.forward_decoder(encoder_output, predicted_sequence.long(), cache=cache)
            predicted_token = torch.argmax(predictions[:, :, -1]).item()
            predicted_sequence = torch.cat([predicted_sequence, torch.argmax(predictions[:, :, -1], dim=1, keepdim=True)], dim=1)
            if predicted_token == self.w2i[EOT_TOKEN]:
                break
        
        dec = [self.i2w[token.item()] for token in predicted_sequence.squeeze(0)[1:]]
        gt = [self.i2w[token.item()] for token in y.squeeze(0)[:-1]]
        
        self.valpredictions.append(dec)
        self.valgts.append(gt)
    
    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)
    
    def get_dictionaries(self):
        return self.w2i, self.i2w


####################################################### POLYPHONY DAN MODEL:

class Poliphony_DAN(DAN):
    def __init__(self, **kwargs):
        super(Poliphony_DAN, self).__init__(**kwargs)
    
    def on_validation_epoch_end(self, name="val"):
        metrics = compute_metrics(self.valgts, self.valpredictions, compute_mv2h=False)
        
        random_index = np.random.randint(0, len(self.valpredictions))
        predtoshow = self.valpredictions[random_index]
        gttoshow = self.valgts[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")

        # Iterate over the keys of the metrics dictionary
        for key in metrics.keys():
            self.log(f'{name}_{key}', metrics[key], prog_bar=True)

        self.valpredictions = []
        self.valgts = []

        return metrics["sym-er"]

    def on_test_epoch_end(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(f"{self.out_dir}/hyp", exist_ok=True)
        os.makedirs(f"{self.out_dir}/gt", exist_ok=True)
        counter = 0
        for sample, gt in zip(self.valpredictions, self.valgts):
            with open(f"{self.out_dir}/hyp/{counter}.krn", "w") as f:
                strsample = [token if token == "<b>" else "\t" + token for token in sample]
                strsample = "".join(strsample)
                strsample = strsample.replace("<b>", "\n")
                f.write(strsample)
            with open(f"{self.out_dir}/gt/{counter}.krn", "w") as f:
                strsample = [token if token == "<b>" else "\t" + token for token in gt]
                strsample = "".join(strsample)
                strsample = strsample.replace("<b>", "\n")
                f.write(strsample)

            counter += 1
        return self.on_validation_epoch_end(name="test")
    
####################################################################################

@gin.configurable
def get_model(in_channels, d_model, dim_ff, max_height, max_width, max_len, out_categories, w2i, i2w, out_dir, h_red=16, w_red=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Poliphony_DAN(in_channels=in_channels, d_model=d_model, 
                          dim_ff=dim_ff, maxh=(max_height//h_red)+1, maxw=(max_width//w_red)+1, 
                          maxlen=max_len+1, out_categories=out_categories, 
                          padding_token=w2i[PAD_TOKEN], w2i=w2i, i2w=i2w, out_dir=out_dir).to(device)
    assert in_channels == NUM_CHANNELS, f"in_channels {in_channels} != NUM_CHANNELS {NUM_CHANNELS}"
    assert max_height == SPECTROGRAM_HEIGHT, f"max_height {max_height} != SPECTROGRAM_HEIGHT {SPECTROGRAM_HEIGHT}"
    summary(model, input_size=[(1, in_channels, max_height, max_width), (1, max_len)], dtypes=[torch.float, torch.long])
    return model
