import torch
import torch.nn.functional as F
import random
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange

# The implementation of the MINiT model is based on the following paper:
# https://arxiv.org/abs/2208.09567


class Head(nn.Module):
    def __init__(
        self,
        *,
        block_size,
        image_size,
        num_classes,
        amp_enabled=False,
        fusion="NA",
        **kwargs,
    ):
        super().__init__()
        block_count = image_size // block_size
        self.fusion = fusion
        self.linear = nn.Linear(
            block_count**3 * num_classes, num_classes if num_classes > 2 else 1
        )
        self.amp_enabled = amp_enabled

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            logits = self.linear(x)

        return logits


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, modality=None):
        super().__init__()
        self.modality = modality
        self.norm = nn.LayerNorm(dim)
        if self.modality == "multi":
            self.norm_b = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x_a, x_b=None, **kwargs):
        if self.modality == "multi":
            return self.fn(self.norm(x_a), self.norm(x_b), **kwargs)
        else:
            return self.fn(self.norm(x_a), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, modality, dim, heads=8, dim_head=64, dropout=0.0, drophead=0, bias=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.modality = modality

        if self.modality == "multi":
            self.to_q = nn.Linear(dim, inner_dim, bias=bias)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
            self.to_qkv = [self.to_q, self.to_kv]

        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)

        self.attention = nn.Identity()

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.drophead = drophead

    def forward(self, x, x_b=None, mask=None):
        _b, _n, _, h = *x.shape, self.heads

        if self.modality == "multi":
            q = self.to_qkv[0](x)
            kv = self.to_qkv[1](x_b).chunk(2, dim=-1)
            qkv = torch.cat((q, kv[0], kv[1]), dim=-1).chunk(3, dim=-1)
        else:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        attn = torch.where(
            attn > 0.01, attn, torch.zeros_like(attn)
        )  # thresholding the attention weights

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        modality,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        drophead=0,
        layerdrop=0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    modality,
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                    drophead=drophead,
                                ),
                                modality=modality,
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )
        self.layerdrop = layerdrop
        self.drophead = drophead

    def forward(self, x_a, x_b=None, mask=None):
        drop_locations = []  # these are the indices to drop.
        if self.drophead != 0:
            if self.training:
                for i in range(len(self.layers)):
                    if random.random() < self.drophead:
                        drop_locations.append(i)
            else:
                # For evaluation, use the "drop every other layer strategy" outlined in
                # https://arxiv.org/pdf/1909.11556.pdf
                cur = 1
                while cur < len(self.layers):
                    drop_locations.append(cur)
                    cur += 1 / self.drophead

        # Different sort of pruning
        x = x_a
        for i, (attn, ff) in enumerate(self.layers):
            # if i in drop_locations:
            #     continue
            x = attn(x, x_b=x_b, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        modality,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        drophead=0,
        layerdrop=0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size."
        )
        assert layerdrop == 0 or int(1 / layerdrop) == 1 / layerdrop, (
            "1/layerdrop needs to be an integer"
        )
        num_patches = (image_size // patch_size) ** 3
        patch_dim = channels * patch_size**3
        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )

        self.to_patch_embedding_a = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (l p3) -> b (h w l) (p1 p2 p3 c)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding_a = nn.Parameter(
            torch.randn(1, num_patches + 1, dim)
        )  # 1, 4**3 + 1, 512
        self.cls_token_a = nn.Parameter(torch.randn(1, 1, dim))

        self.modality = modality
        if self.modality == "multi":
            self.to_patch_embedding_b = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) (l p3) -> b (h w l) (p1 p2 p3 c)",
                    p1=patch_size,
                    p2=patch_size,
                    p3=patch_size,
                ),
                nn.Linear(patch_dim, dim),
            )
            self.pos_embedding_b = nn.Parameter(
                torch.randn(1, num_patches + 1, dim)
            )  # 1, 4**3 + 1, 512
            self.cls_token_b = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            modality,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            drophead=drophead,
            layerdrop=layerdrop,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    # block_embedding is the positional embedding for the block.
    def forward(self, img_a, img_b=None, mask=None, block_embedding=None):
        x_a = self.to_patch_embedding_a(img_a)
        b, n, _ = x_a.shape

        cls_token_a = repeat(self.cls_token_a, "() n d -> b n d", b=b)
        x_a = torch.cat((cls_token_a, x_a), dim=1)
        x_a += self.pos_embedding_a[:, : (n + 1)]
        x_a += block_embedding
        x_a = self.dropout(x_a)

        if self.modality == "multi":
            x_b = self.to_patch_embedding_b(img_b)
            cls_token_b = repeat(self.cls_token_b, "() n d -> b n d", b=b)
            x_b = torch.cat((cls_token_b, x_b), dim=1)
            x_b += self.pos_embedding_b[:, : (n + 1)]
            x_b += block_embedding
            x_b = self.dropout(x_b)
        else:
            x_b = None

        x = self.transformer(x_a, x_b, mask)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class MINiT(nn.Module):
    def __init__(
        self,
        *,
        block_size,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        drophead=0,
        layerdrop=0,
        amp_enabled=False,
        **kwargs,
    ):
        super().__init__()

        self.modality = kwargs["modality"]
        self.image_size = image_size
        self.block_size = block_size
        self.block_count = (
            self.image_size // self.block_size
        )  # block count per side (block_count**3 total blocks)
        self.channels = channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.dim = dim

        self.vit = ViT(
            modality=self.modality,
            image_size=block_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            drophead=drophead,
            layerdrop=layerdrop,
        )

        if self.modality == "multi":
            self.vit_rev = ViT(
                modality=self.modality,
                image_size=block_size,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                pool=pool,
                channels=channels,
                dim_head=dim_head,
                dropout=dropout,
                emb_dropout=emb_dropout,
                drophead=drophead,
                layerdrop=layerdrop,
            )

        self.block_embeddings = nn.Parameter(
            torch.randn(self.block_count**3, (block_size // patch_size) ** 3 + 1, dim)
        )
        self.amp_enabled = amp_enabled

    def forward(self, img_a, img_b=None):
        b = img_a.shape[0]
        p = self.block_size  # this is side length
        block_count = self.block_count
        x_a = rearrange(
            img_a, "b c (h p1) (w p2) (l p3) -> (b h w l) c p1 p2 p3", p1=p, p2=p, p3=p
        )

        if img_b is not None:
            x_b = rearrange(
                img_b,
                "b c (h p1) (w p2) (l p3) -> (b h w l) c p1 p2 p3",
                p1=p,
                p2=p,
                p3=p,
            )
        else:
            x_b = None

        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            # bi-attention for [MRI PET]
            results = self.vit(
                x_a, x_b, block_embedding=self.block_embeddings.repeat(b, 1, 1)
            ).float()
            results = rearrange(
                results,
                "(b h w l) n ->  b (h w l n)",
                h=block_count,
                w=block_count,
                l=block_count,
                n=self.num_classes,
            )

            # bi-attention for [PET MRI]
            if self.modality == "multi":
                results_rev = self.vit_rev(
                    x_b, x_a, block_embedding=self.block_embeddings.repeat(b, 1, 1)
                ).float()
                results_rev = rearrange(
                    results_rev,
                    "(b h w l) n ->  b (h w l n)",
                    h=block_count,
                    w=block_count,
                    l=block_count,
                    n=self.num_classes,
                )
                results = results + results_rev

        return results


class DiaMond:
    def __init__(self):
        self.probe_mri = None  # Placeholder for the MRI probe tensor
        self.probe_mri_linear = None  # Placeholder for the linear layer for the MRI probe
        self.probe_pet = None  # Placeholder for the PET probe tensor
        self.probe_pet_linear = None  # Placeholder for the linear layer for the PET probe
        self.train_with_probes = False  # Default to not using probes for missing pairs

    def body(self, **kwargs):
        return MINiT(**kwargs)

    def body_all(self, PATH_PET=None, PATH_MRI=None, **kwargs):
        kwargs["modality"] = "mono_pet"
        model_pet = self.body(**kwargs)
        if PATH_PET is not None:
            self.load(model_pet, PATH_PET)
            # model_pet.eval()

        kwargs["modality"] = "mono_mri"
        model_mri = self.body(**kwargs)
        if PATH_MRI is not None:
            self.load(model_mri, PATH_MRI)
            # model_mri.eval()

        # Initialize the MRI probe tensor and its linear layer
        if self.probe_mri is None:
            k = kwargs.get("probe_size", 8)  # Default size of the MRI probe tensor
            dim = kwargs["dim"]
            self.probe_mri = nn.Parameter(torch.ones(1, 1, k, k, k))  # Trainable tensor
            self.probe_mri_linear = nn.Linear(k**3, dim)

        # Initialize the PET probe tensor and its linear layer
        if self.probe_pet is None:
            k = kwargs.get("probe_size", 8)  # Default size of the PET probe tensor
            dim = kwargs["dim"]
            self.probe_pet = nn.Parameter(torch.ones(1, 1, k, k, k))  # Trainable tensor
            self.probe_pet_linear = nn.Linear(k**3, dim)

        kwargs["modality"] = "multi"
        model_mp = self.body(**kwargs)
        return model_pet, model_mri, model_mp

    def body_mp(self, PATH_PET=None, PATH_MRI=None, **kwargs):
        kwargs["modality"] = "multi"
        model_mp = self.body(**kwargs)
        return model_mp

    def head(self, **kwargs):
        return Head(**kwargs)

    def save(self, model, PATH):
        torch.save(model.state_dict(), PATH)

    def load(self, model, PATH):
        msg = model.load_state_dict(torch.load(PATH)["model_state_dict"][0])
        print("Loaded model from: ", PATH)
        print(msg)


if __name__ == "__main__":
    PET = torch.ones(1, 1, 128, 128, 128)
    MRI = torch.ones(1, 1, 128, 128, 128)

    diamond = DiaMond()
    PATH_PET, PATH_MRI = None, None  #'/path/to/pet/model.pt', '/path/to/mri/model.pt'

    model_pet, model_mri, model_mp = diamond.body_all(
        PATH_PET,
        PATH_MRI,
        modality="multi",
        block_size=32,
        image_size=128,
        patch_size=8,
        num_classes=2,
        channels=1,
        dim=512,
        depth=4,
        heads=8,
        mlp_dim=309,
    )

    head = diamond.head(
        block_size=32,
        image_size=128,
        num_classes=2,
        channels=1,
    )

    latent_pet = model_pet(PET)
    latent_mri = model_mri(MRI)
    latent_mp = model_mp(PET, MRI)
    preds = head(latent_mp)

    print("Multi modality: ", latent_pet.shape, latent_mri.shape, latent_mp.shape)
    print("Multi modality preds: ", preds.shape)
