class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.cbam1 = CBAM(in_channels)
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.cbam2 = CBAM(in_channels+skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.cbam3 = CBAM(out_channels)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x, skip=None):
        x=self.cbam1(x)
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x=self.cbam2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x=self.cbam3(x)
        return x
