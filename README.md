# NLP

## 環境構築

Poetryを使用します。

1. Poetryのインストール
```bash
# Windows
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
# Mac
curl -sSL https://install.python-poetry.org | python -
echo 'export PATH="/Users/satoru/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```
2. 設定を変更する

```bash
poetry config --list
poetry config virtualenvs.in-project true # プロジェクトフォルダ内に.venvディレクトリを作成する
```
3. Python & ライブラリインストール

```bash
poetry install
```

## ファインチューニング検証

### 目的

- GPTが担っていたタスクの一部を別の言語モデルに切り出すことで、処理の高速化(小規模のモデルの方が結果を返すまでの時間が短い)・API利用料の削減を実現したい
- GPTの入力と出力の対を学習データとして小規模言語モデルをファインチューニングする(知識蒸留)ことで、GPTと遜色のない出力ができるようにモデルを訓練できるかを検証する
- 評価方法はタスク依存だが、基本はGPTとどの程度出力が近いか？を判断基準とする。現状GPTにやらせているタスク自体、どのように評価するかが曖昧なため応急処置的な評価方法である点に注意。

### 手順

1. タスクの設定

2. タスク学習用・評価用のデータセットを準備

    - 学習用：(GPTを使い)自動で作成
    - 評価用：手動or半手動で作成

    - 必要データ量

3. ファインチューニング: `fine-tuning.ipynb`

4. 評価の実施: `evaluation.ipynb`



|タスク|学習データ|評価データ|結果||
|---|---|---|---|---|
|感情分析||||`sentiment_classification`|
|固有名詞判定(固有表現抽出)||||`proper_noun_detection`|
|人名マスキング||||-|
|商品タグ付け||||-|
|サービス・店舗タグ付け||||-|

### モデル選択

- [日本語言語理解ベンチマークJGLUEの構築 〜 自然言語処理モデルの評価用データセットを公開しました](https://techblog.yahoo.co.jp/entry/2022122030379907/)

### ファインチューニング方法

- [huggingfaceのTrainerクラスを使えばFineTuningの学習コードがスッキリ書けてめちゃくちゃ便利です](https://qiita.com/m__k/items/2c4e476d7ac81a3a44af)
- [Hugging Face謹製のTrainerが結構便利というお話](https://qiita.com/tealgreen0503/items/246b7e15e2962b6f9c2b)