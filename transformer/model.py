import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.nn import CrossEntropyLoss
from torchinfo import summary

from data.ar_dataset import EOS_TOKEN, SOS_TOKEN
from data.preprocessing import NUM_CHANNELS
from transformer.decoder import Decoder
from transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION, Encoder
from utils.metrics import compute_metrics


class PositionalEncoding2D(nn.Module):
    """
    Positional Encoding for 3D data (2D spatial + 1D channel).

    Args:
        num_channels (int): Number of channels.
        max_height (int): Maximum height.
        max_width (int): Maximum width.
        dropout_p (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self, num_channels: int, max_height: int, max_width: int, dropout_p: float = 0.1
    ):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos_h = torch.arange(max_height).unsqueeze(1)
        pos_w = torch.arange(max_width).unsqueeze(1)
        den = torch.pow(10000, torch.arange(0, num_channels // 2, 2) / num_channels)

        pe = torch.zeros(1, max_height, max_width, num_channels)
        pe[0, :, :, 0 : num_channels // 2 : 2] = (
            torch.sin(pos_w / den).unsqueeze(0).repeat(max_height, 1, 1)
        )
        pe[0, :, :, 1 : num_channels // 2 : 2] = (
            torch.cos(pos_w / den).unsqueeze(0).repeat(max_height, 1, 1)
        )
        pe[0, :, :, num_channels // 2 :: 2] = (
            torch.sin(pos_h / den).unsqueeze(1).repeat(1, max_width, 1)
        )
        pe[0, :, :, (num_channels // 2) + 1 :: 2] = (
            torch.cos(pos_h / den).unsqueeze(1).repeat(1, max_width, 1)
        )
        pe = pe.permute(0, 3, 1, 2).contiguous()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, num_channels, h, w]
        x = x + self.pe[:, :, : x.size(2), : x.size(3)]
        return self.dropout(x)


##################################################################### UNIMODAL TRANSFORMER:


class Transformer(LightningModule):
    """
    Transformer model for unimodal music transcription.

    Args:
        max_input_height (int): Maximum input height.
        max_input_width (int): Maximum input width.
        max_seq_len (int): Maximum sequence length.
        w2i (Dict[str, int]): Word to index dictionary.
        i2w (Dict[int, str]): Index to word dictionary.
        ytest_i2w (Optional[Dict[int, str]], optional): Index to word dictionary for test set. Defaults to None.
        attn_window (int, optional): Attention window size (number of past tokens to attend to). Defaults to -1.
        teacher_forcing_prob (float, optional): Probability of applying teacher forcing. Defaults to 0.5.
    """

    def __init__(
        self,
        max_input_height: int,
        max_input_width: int,
        max_seq_len: int,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        ytest_i2w: Optional[Dict[int, str]] = None,
        attn_window: int = -1,
        teacher_forcing_prob: float = 0.5,
    ):
        super(Transformer, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        self.padding_idx = w2i["<PAD>"]
        # Model
        self.max_seq_len = max_seq_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.encoder = Encoder(in_channels=NUM_CHANNELS)
        self.pos_2d = PositionalEncoding2D(
            num_channels=256,
            max_height=math.ceil(max_input_height / HEIGHT_REDUCTION),
            max_width=math.ceil(max_input_width / WIDTH_REDUCTION),
        )
        self.decoder = Decoder(
            output_size=len(self.w2i),
            max_seq_len=max_seq_len,
            num_embeddings=len(self.w2i),
            padding_idx=self.padding_idx,
            attn_window=attn_window,
        )
        self.summary(max_input_height, max_input_width)
        # Loss
        self.compute_loss = CrossEntropyLoss(ignore_index=self.padding_idx)
        # Predictions
        self.Y = []
        self.YHat = []

    def summary(self, max_input_height: int, max_input_width: int):
        print("Encoder")
        summary(
            self.encoder,
            input_size=[1, NUM_CHANNELS, max_input_height, max_input_width],
        )
        print("Decoder")
        tgt_size = [1, self.max_seq_len]
        memory_size = [
            1,
            math.ceil(max_input_height / HEIGHT_REDUCTION)
            * math.ceil(max_input_width / WIDTH_REDUCTION),
            256,
        ]
        memory_len_size = [1]
        summary(
            self.decoder,
            input_size=[tgt_size, memory_size, memory_len_size],
            dtypes=[torch.int64, torch.float32, torch.int64],
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-4,
            amsgrad=False,
        )

    def forward(
        self, x: torch.Tensor, xl: torch.Tensor, y_in: torch.Tensor
    ) -> torch.Tensor:
        # Encoder
        x = self.encoder(x=x)
        # Prepare for decoder
        # 2D PE + flatten + permute
        x = self.pos_2d(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        # Decoder
        y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl)
        return y_out_hat

    def apply_teacher_forcing(self, y: torch.Tensor) -> torch.Tensor:
        """Error the ground truth sequence with a probability of `teacher_forcing_prob`."""
        # y.shape = [batch_size, seq_len]
        y_errored = y.clone()
        for i in range(y_errored.size(0)):
            for j in range(y_errored.size(1)):
                if (
                    random.random() < self.teacher_forcing_prob
                    and y[i, j] != self.padding_idx
                ):
                    y_errored[i, j] = random.randint(0, len(self.w2i) - 1)
        return y_errored

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, xl, y_in, y_out = batch
        y_in = self.apply_teacher_forcing(y_in)
        yhat = self.forward(x=x, xl=xl, y_in=y_in)
        loss = self.compute_loss(yhat, y_out)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        assert x.size(0) == y.size(0) == 1, "Inference only supports batch_size = 1"

        # Encoder
        x = self.encoder(x=x)
        # Prepare for decoder
        # 2D PE + flatten + permute
        x = self.pos_2d(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        # Autoregressive decoding
        y_in = torch.tensor([self.w2i[SOS_TOKEN]]).unsqueeze(0).long().to(x.device)
        yhat = []
        for _ in range(self.max_seq_len):
            y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=None)
            y_out_hat = y_out_hat[0, :, -1]  # Last token
            y_out_hat_token = y_out_hat.argmax(dim=-1).item()
            y_out_hat_word = self.i2w[y_out_hat_token]
            yhat.append(y_out_hat_word)
            if y_out_hat_word == EOS_TOKEN:
                break

            y_in = torch.cat(
                [y_in, torch.tensor([[y_out_hat_token]]).long().to(x.device)], dim=1
            )

        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0][1:]]  # Remove SOS_TOKEN
        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(
        self, name: str = "val", print_random_samples: bool = False
    ):
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)

    ##### FOR LATE MULTIMODAL FUSION:

    def get_pred_seq_and_pred_prob_seq(
        self, x: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """
        Returns the predicted sequence and the probability of each predicted token for a given input.
        Useful for Smith-Waterman late-multimodal fusion.

        Args:
            x (torch.Tensor): Input tensor of shape [1, NUM_CHANNELS, H, W].

        Returns:
            Tuple[List[str], List[float]]: The predicted sequence and the probability of each predicted token.
        """
        assert x.size(0) == 1, "Inference only supports batch_size = 1"

        # Encoder
        x = self.encoder(x=x)
        # Prepare for decoder
        # 2D PE + flatten + permute
        x = self.pos_2d(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        # Autoregressive decoding
        y_in = torch.tensor([self.w2i[SOS_TOKEN]]).unsqueeze(0).long().to(x.device)
        yhat = []
        yhat_prob = []
        for _ in range(self.max_seq_len):
            y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=None)
            y_out_hat = y_out_hat[0, :, -1]  # Last token
            y_out_hat_prob, y_out_hat_token = y_out_hat.topk(k=1, dim=-1)
            y_out_hat_word = self.i2w[y_out_hat_token.item()]
            yhat.append(y_out_hat_word)
            yhat_prob.append(y_out_hat_prob.item())
            if y_out_hat_word == EOS_TOKEN:
                break

            y_in = torch.cat(
                [y_in, y_out_hat_token.unsqueeze(0).long().to(x.device)], dim=1
            )

        return yhat, yhat_prob


##################################################################### MULTIMODAL TRANSFORMER:


class CrossAttention(nn.Module):
    """
    Perform cross-attention between two sequences.

    Args:
        feature_dim (int): Dimension of the input features. The same for query, key, and value.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Methods:
        forward(query: torch.Tensor, len_query: Optional[torch.Tensor], key_value: torch.Tensor, len_key_value: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]: Forward pass.
            - query: Query tensor of shape [B, len_a, feature_dim].
            - len_query: Tensor of shape [B] containing the actual length of each sequence in query tensor. It is used to create the attention mask. If None, no mask is applied.
            - key_value: Key and Value tensor of shape [B, len_b, feature_dim]. It is the same tensor for both key and value.
            - len_key_value: Tensor of shape [B] containing the actual length of each sequence in key_value tensor. It is used to create the attention mask. If None, no mask is applied.

            Returns:
            - attn_output: Output tensor of shape [B, len_a, feature_dim].
            - attn_weights: Attention weights tensor of shape [B, len_a, len_b].
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        len_query: Optional[torch.Tensor],
        key_value: torch.Tensor,
        len_key_value: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # query.shape = [B, len_a, feature_dim]
        # len_query.shape = [B]; contains the actual length of each sequence in query
        # key_value.shape = [B, len_b, feature_dim]
        # len_key_value.shape = [B]; contains the actual length of each sequence in key_value

        # Compute the attention mask
        if len_query is not None and len_key_value is not None:
            attn_mask = self.create_attention_mask(
                query=query,
                len_query=len_query,
                key_value=key_value,
                len_key_value=len_key_value,
            )
        else:
            attn_mask = None

        # Apply multi-head attention
        attn_output, attn_weights = self.attention(
            query=query, key=key_value, value=key_value, attn_mask=attn_mask
        )
        # attn_output.shape = [B, len_a, feature_dim]
        # attn_weights.shape = [B, len_a, len_b]

        return attn_output, attn_weights

    def create_attention_mask(
        self,
        query: torch.Tensor,
        len_query: torch.Tensor,
        key_value: torch.Tensor,
        len_key_value: torch.Tensor,
    ) -> torch.Tensor:
        # query.shape = [B, len_a, feature_dim]
        # len_query.shape = [B]; contains the actual length of each sequence in query
        # key_value.shape = [B, len_b, feature_dim]
        # len_key_value.shape = [B]; contains the actual length of each sequence in key_value

        # Create the attention mask -> [B, len_a, len_b] to allow for a different mask for each input in the batch
        # For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
        attn_mask = torch.zeros(
            query.shape[0],
            query.shape[1],
            key_value.shape[1],
            device=query.device,
        )
        for id, (lq, lkv) in enumerate(zip(len_query, len_key_value)):
            attn_mask[id, lq:, lkv:] = 1
        attn_mask = attn_mask.bool()

        # Repeat the mask for each head -> [B * num_heads, len_a, len_b]
        attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        return attn_mask


class MultimodalTransformer(LightningModule):
    """
    Transformer model for multimodal music transcription.

    Args:
        max_img_height (int): Maximum image height.
        max_img_width (int): Maximum image width.
        max_audio_height (int): Maximum audio height.
        max_audio_width (int): Maximum audio width.
        max_seq_len (int): Maximum sequence length.
        w2i (Dict[str, int]): Word to index dictionary.
        i2w (Dict[int, str]): Index to word dictionary.
        ytest_i2w (Optional[Dict[int, str]], optional): Index to word dictionary for test set. Defaults to None.
        mixer_type (str, optional): Modality mixer type. Defaults to "concat".
        attn_window (int, optional): Attention window size (number of past tokens to attend to). Defaults to -1.
        teacher_forcing_prob (float, optional): Probability of applying teacher forcing. Defaults to 0.5.
        teacher_forcing_modality_prob (float, optional): Probability of using both modalities or only one. Defaults to 0.5.
    """

    def __init__(
        self,
        max_img_height: int,
        max_img_width: int,
        max_audio_height: int,
        max_audio_width: int,
        max_seq_len: int,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        ytest_i2w: Optional[Dict[int, str]] = None,
        mixer_type: str = "concat",
        attn_window: int = -1,
        teacher_forcing_prob: float = 0.5,
        teacher_forcing_modality_prob: float = 0.5,
    ):
        super(MultimodalTransformer, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        self.padding_idx = w2i["<PAD>"]
        # Model
        self.max_seq_len = max_seq_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.teacher_forcing_modality_prob = teacher_forcing_modality_prob
        self.image_encoder = Encoder(in_channels=NUM_CHANNELS)
        self.image_pos_2d = PositionalEncoding2D(
            num_channels=256,
            max_height=math.ceil(max_img_height / HEIGHT_REDUCTION),
            max_width=math.ceil(max_img_width / WIDTH_REDUCTION),
        )
        self.audio_encoder = Encoder(in_channels=NUM_CHANNELS)
        self.audio_pos_2d = PositionalEncoding2D(
            num_channels=256,
            max_height=math.ceil(max_audio_height / HEIGHT_REDUCTION),
            max_width=math.ceil(max_audio_width / WIDTH_REDUCTION),
        )
        self.decoder = Decoder(
            output_size=len(self.w2i),
            max_seq_len=max_seq_len,
            num_embeddings=len(self.w2i),
            padding_idx=self.padding_idx,
            attn_window=attn_window,
        )
        # Set modality mixer
        if mixer_type == "concat":
            self.mixer = self.mixer_concat
        elif mixer_type == "attn_img":
            self.cross_attn = CrossAttention(feature_dim=256)
            self.mixer = self.mixer_attn_img
        elif mixer_type == "attn_audio":
            self.cross_attn = CrossAttention(feature_dim=256)
            self.mixer = self.mixer_attn_audio
        elif mixer_type == "attn_both":
            self.cross_attn = CrossAttention(feature_dim=256)
            self.mixer = self.mixer_attn_both
        else:
            raise ValueError(f"Invalid mixer type: {mixer_type}")
        self.summary(max_img_height, max_img_width, max_audio_height, max_audio_width)
        # Loss
        self.compute_loss = CrossEntropyLoss(ignore_index=self.padding_idx)
        # Predictions
        self.Y = []
        self.YHat = []

    def summary(
        self,
        max_img_height: int,
        max_img_width: int,
        max_audio_height: int,
        max_audio_width: int,
    ):
        print("Image Encoder")
        summary(
            self.image_encoder,
            input_size=[1, NUM_CHANNELS, max_img_height, max_img_width],
        )
        print("Audio Encoder")
        summary(
            self.audio_encoder,
            input_size=[1, NUM_CHANNELS, max_audio_height, max_audio_width],
        )
        print("Decoder")
        tgt_size = [1, self.max_seq_len]
        memory_size = [
            1,
            math.ceil(max(max_img_height, max_audio_height) / HEIGHT_REDUCTION)
            * math.ceil(max(max_img_width, max_audio_width) / WIDTH_REDUCTION),
            256,
        ]
        memory_len_size = [1]
        summary(
            self.decoder,
            input_size=[tgt_size, memory_size, memory_len_size],
            dtypes=[torch.int64, torch.float32, torch.int64],
        )

    def configure_optimizers(self):
        params = (
            list(self.image_encoder.parameters())
            + list(self.audio_encoder.parameters())
            + list(self.decoder.parameters())
        )
        if hasattr(self, "cross_attn"):
            params += list(self.cross_attn.parameters())
        return torch.optim.Adam(
            params,
            lr=1e-4,
            amsgrad=False,
        )

    def encoder_forward(
        self,
        xi: torch.Tensor,
        xa: torch.Tensor,
        xli: Optional[torch.Tensor] = None,
        xla: Optional[torch.Tensor] = None,
        apply_teacher_forcing_modality: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Image encoder
        xi = self.image_encoder(xi)
        # Prepare for decoder
        # 2D PE + flatten + permute
        xi = self.image_pos_2d(xi)
        xi = xi.flatten(2).permute(0, 2, 1).contiguous()
        # xi.shape -> [B, C, H, W] -> [B, H*W, C] -> [B, LenImage, C]

        # Audio encoder
        xa = self.audio_encoder(xa)
        # Prepare for decoder
        # 2D PE + flatten + permute
        xa = self.audio_pos_2d(xa)
        xa = xa.flatten(2).permute(0, 2, 1).contiguous()
        # xa.shape -> [B, C, H, W] -> [B, H*W, C] -> [B, LenAudio, C]

        # Combine them
        if apply_teacher_forcing_modality:
            modality = self.apply_teacher_forcing_modality()
            if modality == "image":
                return xi, xli
            elif modality == "audio":
                return xa, xla
            elif modality == "both":
                x, xl = self.mixer(xi=xi, xa=xa, xli=xli, xla=xla)
            else:
                raise ValueError(f"Invalid modality: {modality}")
        else:
            x, xl = self.mixer(xi=xi, xa=xa, xli=xli, xla=xla)
        return x, xl

    def forward(
        self,
        xi: torch.Tensor,
        xli: torch.Tensor,
        xa: torch.Tensor,
        xla: torch.Tensor,
        y_in: torch.Tensor,
        apply_teacher_forcing_modality: bool = False,
    ) -> torch.Tensor:
        # Encoder
        x, xl = self.encoder_forward(
            xi=xi,
            xa=xa,
            xli=xli,
            xla=xla,
            apply_teacher_forcing_modality=apply_teacher_forcing_modality,
        )
        # Decoder
        y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl)
        return y_out_hat

    def apply_teacher_forcing(self, y: torch.Tensor) -> torch.Tensor:
        """Error the ground truth sequence with a probability of `teacher_forcing_prob`."""
        # y.shape = [batch_size, seq_len]
        y_errored = y.clone()
        for i in range(y_errored.size(0)):
            for j in range(y_errored.size(1)):
                if (
                    random.random() < self.teacher_forcing_prob
                    and y[i, j] != self.padding_idx
                ):
                    y_errored[i, j] = random.randint(0, len(self.w2i) - 1)
        return y_errored

    def apply_teacher_forcing_modality(self) -> str:
        """
        Choose one modality at random with a probability of `teacher_forcing_modality_prob`.

        Returns:
            str: Chosen modality. Can be "image", "audio", or "both".
        """
        if random.random() < self.teacher_forcing_modality_prob:
            # Chose one modality at random
            if random.random() < 0.5:
                return "image"
            else:
                return "audio"
        else:
            return "both"

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        xi, xli, xa, xla, y_in, y_out = batch
        y_in = self.apply_teacher_forcing(y_in)
        yhat = self.forward(
            xi=xi,
            xli=xli,
            xa=xa,
            xla=xla,
            y_in=y_in,
            apply_teacher_forcing_modality=True,
        )
        loss = self.compute_loss(yhat, y_out)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xi, xa, y = batch
        assert (
            xi.size(0) == xa.size(0) == y.size(0) == 1
        ), "Inference only supports batch_size = 1"

        # Encoder
        x, _ = self.encoder_forward(
            xi=xi, xa=xa, xli=None, xla=None, apply_teacher_forcing_modality=False
        )
        # Autoregressive decoding
        y_in = torch.tensor([self.w2i[SOS_TOKEN]]).unsqueeze(0).long().to(x.device)
        yhat = []
        for _ in range(self.max_seq_len):
            y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=None)
            y_out_hat = y_out_hat[0, :, -1]  # Last token
            y_out_hat_token = y_out_hat.argmax(dim=-1).item()
            y_out_hat_word = self.i2w[y_out_hat_token]
            yhat.append(y_out_hat_word)
            if y_out_hat_word == EOS_TOKEN:
                break

            y_in = torch.cat(
                [y_in, torch.tensor([[y_out_hat_token]]).long().to(x.device)], dim=1
            )

        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0][1:]]  # Remove SOS_TOKEN
        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(
        self, name: str = "val", print_random_samples: bool = False
    ):
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)

    ##### MODALITY MIXERS:

    def mixer_concat(
        self,
        xi: torch.Tensor,
        xa: torch.Tensor,
        xli: Optional[torch.Tensor] = None,
        xla: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Concatenate the image and audio features."""
        # xi.shape = [B, LenImage, C]
        # xa.shape = [B, LenAudio, C]
        x = torch.cat([xi, xa], dim=1)  # [B, LenImage + LenAudio, C]
        if xli is not None and xla is not None:
            xl = xli + xla
        else:
            xl = None
        return x, xl

    def mixer_attn_img(
        self,
        xi: torch.Tensor,
        xa: torch.Tensor,
        xli: Optional[torch.Tensor] = None,
        xla: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend to the image features (key, value) using the audio features (query)."""
        # xi.shape = [B, LenImage, C]
        # xa.shape = [B, LenAudio, C]
        x, _ = self.cross_attn(query=xa, len_query=xla, key_value=xi, len_key_value=xli)
        # x.shape = [B, LenAudio, C]
        if xli is not None and xla is not None:
            xl = xla
        else:
            xl = None
        return x, xl

    def mixer_attn_audio(
        self,
        xi: torch.Tensor,
        xa: torch.Tensor,
        xli: Optional[torch.Tensor] = None,
        xla: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend to the audio features (key, value) using the image features (query)."""
        # xi.shape = [B, LenImage, C]
        # xa.shape = [B, LenAudio, C]
        x, _ = self.cross_attn(query=xi, len_query=xli, key_value=xa, len_key_value=xla)
        # x.shape = [B, LenImage, C]
        if xli is not None and xla is not None:
            xl = xli
        else:
            xl = None
        return x, xl

    def mixer_attn_both(
        self,
        xi: torch.Tensor,
        xa: torch.Tensor,
        xli: Optional[torch.Tensor] = None,
        xla: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend to both the image and audio features using both modalities. Then, concatenate the results."""
        # xi.shape = [B, LenImage, C]
        # xa.shape = [B, LenAudio, C]
        xa, xla = self.mixer_attn_img(xi=xi, xa=xa, xli=xli, xla=xla)
        xi, xli = self.mixer_attn_audio(xi=xi, xa=xa, xli=xli, xla=xla)
        x, xl = self.mixer_concat(xi=xi, xa=xa, xli=xli, xla=xla)
        return x, xl
