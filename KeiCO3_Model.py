 

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from torch.utils.data import Dataset, DataLoader



import pandas as pd
df = pd.read_table('KeiCO_平行.csv', header=0, sep=',')
df_input = df['レベル4'].tolist()
df_target = df['レベル1'].tolist()



class KeicoDataset(Dataset):
    def __init__(self, tokenizer, df_input, df_target):        
        self.df_input = df_input
        self.df_target = df_target
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def _build(self):
        for i in range(len(self.df_input)):
            if type(self.df_input[i]) == str and type(self.df_target[i]) == str:
                tokenized_inputs = tokenizer.batch_encode_plus(
                    [self.df_input[i]], max_length=64, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = tokenizer.batch_encode_plus(
                    [self.df_target[i]], max_length=64, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)
dataset = KeicoDataset(tokenizer, df_input, df_target)



for data in dataset:
    print("A. 入力データの元になる文字列")
    print(tokenizer.decode(data["source_ids"]))
    print()
    print("B. 入力データ（Aの文字列がトークナイズされたトークンID列）")
    print(data["source_ids"])
    print()
    print("C. 出力データの元になる文字列")
    print(tokenizer.decode(data["target_ids"]))
    print()
    print("D. 出力データ（Cの文字列がトークナイズされたトークンID列）")
    print(data["target_ids"])
    break


#################################################################################################################################
import pytorch_lightning as pl


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")

        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked), 
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, df_input, df_target):
        """データセットを作成する"""
        return KeicoDataset(tokenizer, df_input, df_target)
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                             df_input=df_input, df_target=df_target)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                           df_input=df_input, df_target=df_target)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.eval_batch_size, 
                          num_workers=4)


# 各種ハイパーパラメータ
args_dict = dict(
    data_dir=".",  # データセットのディレクトリ
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    n_gpu=1,
    early_stop_callback = False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)


# 学習に用いるハイパーパラメータを設定する
args_dict.update({
    "max_input_length":  64,  # 入力文の最大トークン数
    "max_target_length": 64,  # 出力文の最大トークン数
    "train_batch_size":  80,
    "eval_batch_size":   80,
    "num_train_epochs":  3,
    })
import argparse

args = argparse.Namespace(**args_dict)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
)

# 転移学習の実行（GPUを利用すれば1エポック10分程度）
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

# 最終エポックのモデルを保存
SAVED_MODEL_DIR = "./SAVED_MODEL_DIR"
model.tokenizer.save_pretrained(SAVED_MODEL_DIR)
model.model.save_pretrained(SAVED_MODEL_DIR)

del model

##################################################################
# 学習済みモデルの読み込み
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(SAVED_MODEL_DIR, is_fast=True)

# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(SAVED_MODEL_DIR)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()



import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

# テストデータの読み込み
test_dataset = KeicoDataset(tokenizer, df_input, df_target)
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

trained_model.eval()

inputs = []
outputs = []
targets = []

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    output = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length,
        repetition_penalty=10.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

    output_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False) 
                for ids in output]
    target_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in batch["target_ids"]]
    input_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in input_ids]

    inputs.extend(input_text)
    outputs.extend(output_text)
    targets.extend(target_text)
    
## 生成結果確認

for output, target, input in zip(outputs, targets, inputs):
    print("input of model:  " + input)
    print("output of model: " + output)
    print("teacher data:    " + target)
    print()

## ## 自己打句子进去　## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
Mysentence = ["詳しいことを紹介したいと思うが、明日君の会社に行ってもいいか。"]
test_dataset = KeicoDataset(tokenizer, Mysentence, ["None"])
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

trained_model.eval()

inputs = []
outputs = []
targets = []

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    output = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length,
        repetition_penalty=10.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

    output_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False) 
                for ids in output]
    target_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in batch["target_ids"]]
    input_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in input_ids]

    inputs.extend(input_text)
    outputs.extend(output_text)
    targets.extend(target_text)
    
## 生成結果確認

for output, target, input in zip(outputs, targets, inputs):
    print("input of model:  " + input)
    print("output of model: " + output)
    print("teacher data:    " + target)
    print()