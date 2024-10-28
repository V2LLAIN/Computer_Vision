class U_Net(pl.LightningModule):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        self.jaccard = BinaryJaccardIndex()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',	# Loss최소화,최대화 결정
                                                               factor=1e-8,	# 학습률 감소율
                                                               patience=5,
                                                               verbose=True,)
        monitor_metric = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor_metric  # Specify the metric you want to monitor
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8) # 0아니면 1로 바꾸는
        y = y.to(torch.uint8)
        
        iou = self.jaccard(predicted_masks, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8)
        y = y.to(torch.uint8)
  
        iou = self.jaccard(predicted_masks, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

        return loss


class R2U_Net(pl.LightningModule):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        self.jaccard = BinaryJaccardIndex()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',	# Loss최소화,최대화 결정
                                                               factor=1e-8,	# 학습률 감소율
                                                               patience=5,
                                                               verbose=True,)
        monitor_metric = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor_metric  # Specify the metric you want to monitor
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8) # 0아니면 1로 바꾸는
        y = y.to(torch.uint8)
        
        iou = self.jaccard(predicted_masks, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8)
        y = y.to(torch.uint8)
  
        iou = self.jaccard(predicted_masks, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

        return loss



class AttU_Net(pl.LightningModule):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        self.jaccard = BinaryJaccardIndex()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',	# Loss최소화,최대화 결정
                                                               factor=1e-8,	# 학습률 감소율
                                                               patience=5,
                                                               verbose=True,)
        monitor_metric = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor_metric  # Specify the metric you want to monitor
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8) # 0아니면 1로 바꾸는
        y = y.to(torch.uint8)
        
        iou = self.jaccard(predicted_masks, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8)
        y = y.to(torch.uint8)
  
        iou = self.jaccard(predicted_masks, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

        return loss


class R2AttU_Net(pl.LightningModule):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        self.jaccard = BinaryJaccardIndex()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',	# Loss최소화,최대화 결정
                                                               factor=1e-8,	# 학습률 감소율
                                                               patience=5,
                                                               verbose=True,)
        monitor_metric = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor_metric  # Specify the metric you want to monitor
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8) # 0아니면 1로 바꾸는
        y = y.to(torch.uint8)
        
        iou = self.jaccard(predicted_masks, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['X'], batch['y']
        outputs = self(x).squeeze(dim=1)
        loss = nn.BCELoss()(outputs, y)
        
        predicted_masks = (outputs > 0.5).to(torch.uint8)
        y = y.to(torch.uint8)
  
        iou = self.jaccard(predicted_masks, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

        return loss
